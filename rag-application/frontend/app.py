import os
import time
import uuid
import streamlit as st
from pathlib import Path

# =========================================
#    IMPORTS DO AZURE E LANGCHAIN
# =========================================
os.environ["AZURESEARCH_FIELDS_ID"] = "chunk_id"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "vector"

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from langchain_core.messages import trim_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores.azuresearch import (
    AzureSearch,
    AzureSearchVectorStoreRetriever
)

# =========================================
#    CONFIGURAÇÕES GLOBAIS
# =========================================

st.set_page_config(
    page_title="PoliGPT",
    page_icon="images/poligpt_icon.png",
    layout="centered"  # layout wide para ter espaço + sidebar
)

# Gera ou recupera ID de sessão
if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())

# Caminho para o logo
LOGO_PATH = "images/poligpt_logo.png"

# URL do Key Vault
kv_uri = "https://kv-poligpt-dev-eastus2.vault.azure.net"

# Inicializa credenciais do Azure Key Vault
credential = DefaultAzureCredential()
client = SecretClient(vault_url=kv_uri, credential=credential)

# Buscar chaves/segredos a partir do Key Vault
OPENAI_API_KEY = client.get_secret("openai-api-key").value
AZURE_SEARCH_ENDPOINT = client.get_secret("azure-search-endpoint").value
AZURE_SEARCH_ADMIN_KEY = client.get_secret("azure-search-admin-key").value

# Modelos e parâmetros de inferência
COMPLETION_MODEL = "gpt-4o-2024-08-06"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
MODEL_TEMPERATURE = 0.3

# Configurações de busca
SEARCH_INDEX_NAME = "poligpt-index"
SEARCH_TYPE = "hybrid"
NUM_DOCS_TO_RETRIEVE = 5

# =========================================
#    FUNÇÕES AUXILIARES
# =========================================

@st.cache_resource
def setup_llm(temperature: float = MODEL_TEMPERATURE) -> ChatOpenAI:
    """
    Configura o modelo de linguagem (LLM) com parâmetros específicos.
    """
    llm = ChatOpenAI(
        model=COMPLETION_MODEL,
        api_key=OPENAI_API_KEY,
        streaming=False,
        temperature=temperature
    )
    return llm

@st.cache_resource
def setup_retriever(search_type: str = "hybrid", k: int = 3) -> AzureSearchVectorStoreRetriever:
    """
    Configura o objeto de busca vetorial no Azure Search.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        dimensions=EMBEDDING_DIMENSIONS
    )

    vector_store = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_ADMIN_KEY,
        index_name=SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
        search_type=search_type
    )

    retriever = AzureSearchVectorStoreRetriever(
        name="AzureSearchRetriever",
        vectorstore=vector_store,
        search_type=search_type,
        k=k
    )

    return retriever

def agent_orchestrator(query: str, chat_history: list, session_id: str) -> str:
    """
    Orquestra a chamada ao RAG Agent, incluindo a ferramenta de busca
    no Azure Search, e retorna a resposta final do LLM.
    """
    # Inicializa memória e LLM
    memory = MemorySaver()
    llm = setup_llm(temperature=MODEL_TEMPERATURE)

    # Inicializa retriever
    retriever = setup_retriever(search_type=SEARCH_TYPE, k=NUM_DOCS_TO_RETRIEVE)

    # Define a tool para busca vetorial
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="buscar_poli_info",
        description="Busca e retorna informações sobre a Escola Politécnica da UFRJ."
    )

    # Lista de ferramentas disponíveis para o agente
    tools = [retriever_tool]

    # Mensagem de sistema para direcionar o comportamento do modelo
    system_message = SystemMessage(
        content="""
        Você é o PoliGPT, um assistente virtual da Escola Politécnica da Universidade Federal do Rio de Janeiro (UFRJ),
        que auxilia alunos, professores, funcionários e outros interessados em questões acadêmicas e institucionais 
        relacionadas à Escola Politécnica, além de conhecimentos gerais. Quaisquer perguntas relacionadas à área acadêmica 
        feitas pelo usuário devem ser consideradas como referentes à Escola Politécnica, a não ser que seja explicitamente 
        dito o contrário. Caso não tenha informações suficientes para responder, diga que não sabe. Responda sempre na mesma 
        língua usada pelo usuário; caso não seja possível reconhecer a língua, use português.
        """
    )

    # Cria o agente RAG com as ferramentas definidas
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_message,
        checkpointer=memory
    )

    # Monta a lista de mensagens “processadas”
    processed_history = [system_message]
    for msg in chat_history:
        if msg["role"] == "user":
            processed_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            processed_history.append(AIMessage(content=msg["content"]))

    # Adiciona a última pergunta do usuário
    processed_history.append(HumanMessage(content=query))

    # Remove mensagens antigas se necessário, para não ultrapassar limite de tokens
    trimmed_history = trim_messages(
        processed_history,
        strategy="last",
        token_counter=llm,
        max_tokens=8192,
        start_on="human",
        end_on="human",
        include_system=True,
        allow_partial=False
    )

    config = {"configurable": {"thread_id": session_id}}
    messages = agent_executor.invoke(input={"messages": trimmed_history}, config=config)

    # Pega a resposta final
    output = messages["messages"][-1].content
    return output

# =========================================
#    FUNÇÃO PRINCIPAL DA APLICAÇÃO
# =========================================

def main():
    # 1. SIDEBAR (barra lateral) -----------------------------------------------
    with st.sidebar:
        # Exibe logo se existir
        if Path(LOGO_PATH).is_file():
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown("**PoliGPT**")

        # Exemplo de um menu simples
        st.title("Menu")
        page = st.radio("Ir para:", ["Home"], index=0)

        st.write("---")
        st.write("**Bem-vindo!**")

    # 2. CORPO PRINCIPAL (Home) -----------------------------------------------
    if page == "Home":
        # Título da aplicação (no corpo principal)
        st.title("Seu Assistente Virtual Acadêmico")
        
        # Inicializa histórico de chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Exibe as mensagens do histórico usando os componentes nativos de chat
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                # Supondo que qualquer outra role seja "ai"
                with st.chat_message("assistant"):
                    st.write(msg["content"])

        # Campo de input do usuário (novo componente)
        user_input = st.chat_input("Escreva sua mensagem...")

        if user_input:
            # Adiciona pergunta do usuário ao histórico
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Gera resposta do agente
            start_time = time.time()
            answer = agent_orchestrator(
                query=user_input,
                chat_history=st.session_state.chat_history,
                session_id=st.session_state.id
            )
            end_time = time.time()

            # Adiciona resposta do PoliGPT ao histórico
            st.session_state.chat_history.append(
                {"role": "ai", "content": answer}
            )

            # Reexibe as mensagens (opcionalmente você pode usar st.experimental_rerun)
            with st.chat_message("assistant"):
                st.write(answer)

            # Mostra tempo de execução
            exec_time = end_time - start_time
            st.write(f"*(Tempo de execução: {exec_time:.2f} segundos)*")

# =========================================
#  EXECUÇÃO DO APLICATIVO STREAMLIT
# =========================================
if __name__ == "__main__":
    main()
