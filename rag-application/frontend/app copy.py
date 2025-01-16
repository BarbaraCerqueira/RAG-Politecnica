import os
import time
import uuid
import streamlit as st
from pathlib import Path


# ======================== IMPORTS DO AZURE E LANGCHAIN ========================

os.environ["AZURESEARCH_FIELDS_ID"] = "chunk_id"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "vector"

# ============================ CONFIGURAÇÕES GLOBAIS ============================

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

@st.cache_resource
def get_secrets():
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    # URL do Key Vault
    kv_uri = "https://kv-poligpt-dev-eastus2.vault.azure.net"

    # Inicializa credenciais do Azure Key Vault
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=kv_uri, credential=credential)

    # Buscar chaves/segredos a partir do Key Vault
    OPENAI_API_KEY = client.get_secret("openai-api-key").value
    AZURE_SEARCH_ENDPOINT = client.get_secret("azure-search-endpoint").value
    AZURE_SEARCH_ADMIN_KEY = client.get_secret("azure-search-admin-key").value

    return OPENAI_API_KEY, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY

OPENAI_API_KEY = ''
AZURE_SEARCH_ENDPOINT = ''
AZURE_SEARCH_ADMIN_KEY = ''

# Modelos e parâmetros de inferência
COMPLETION_MODEL = "gpt-4o-2024-08-06"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
MODEL_TEMPERATURE = 0.3

# Configurações de busca
SEARCH_INDEX_NAME = "poligpt-index"
SEARCH_TYPE = "hybrid"
NUM_DOCS_TO_RETRIEVE = 5


# ============================ FUNÇÕES AUXILIARES ============================

@st.cache_resource
def setup_llm(temperature: float = MODEL_TEMPERATURE):
    """
    Configura o modelo de linguagem (LLM) com parâmetros específicos.
    """
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=COMPLETION_MODEL,
        api_key=OPENAI_API_KEY,
        streaming=False,
        temperature=temperature
    )
    return llm

@st.cache_resource
def setup_retriever(search_type: str = "hybrid", k: int = 3):
    """
    Configura o objeto de busca vetorial no Azure Search.
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores.azuresearch import (
        AzureSearch,
        AzureSearchVectorStoreRetriever
    )

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
    from langchain_core.messages import trim_messages
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain.tools.retriever import create_retriever_tool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

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
        que auxilia alunos, professores, funcionários e outros em questões acadêmicas e institucionais relacionadas à Escola Politécnica, 
        além de conhecimentos gerais. Quaisquer perguntas relacionadas à área acadêmica feitas pelo usuário devem ser consideradas 
        como referentes à Escola Politécnica, a não ser que seja explicitamente dito o contrário. Caso não tenha informações suficientes 
        para responder, diga que não sabe. Responda sempre na mesma língua usada pelo usuário; caso não seja possível reconhecer a língua, use português.
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

def show_message(role, content):
    with st.chat_message(role):
        st.write(content)

@st.fragment
def home_page():
    # Título da aplicação (no corpo principal)
    st.title("Seu Assistente Virtual Acadêmico")
    st.write("---")
    
    # Inicializa histórico de chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Exibe as mensagens do histórico usando os componentes nativos de chat
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            show_message(role="user", content=msg["content"])
        else:
            show_message(role="assistant", content=msg["content"])

    # Campo de input do usuário
    user_input = st.chat_input(placeholder="Escreva uma mensagem...")

    if user_input:
        # Adicionar pergunta do usuário ao histórico e exibir
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # show_message(role="user", content=user_input)

        # Gera resposta do agente
        with st.spinner("Gerando resposta..."):
            start_time = time.time()
            answer = agent_orchestrator(
                query=user_input,
                chat_history=st.session_state.chat_history,
                session_id=st.session_state.id
            )
            end_time = time.time()

        # Adicionar resposta do RAG ao histórico e exibir
        st.session_state.chat_history.append({"role": "ai", "content": answer})
        # show_message(role="assistant", content=answer)

        # Mostra tempo de execução
        exec_time = end_time - start_time
        st.write(f"*(Tempo de geração da resposta: {exec_time:.2f} segundos)*")

        # st.rerun(scope='fragment')


# ======================== FUNÇÃO PRINCIPAL DA APLICAÇÃO ========================

def main():
    # 1. SIDEBAR (barra lateral)
    with st.sidebar:
        # Exibe logo se existir
        if Path(LOGO_PATH).is_file():
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown("**PoliGPT**")

        # Exemplo de um menu simples
        st.title("Menu")
        page = st.radio("Ir para:", ["Home", "Settings"], index=0)

        st.write("---")
        st.write("**Bem-vindo!**")

    # 2. CORPO PRINCIPAL (Home)
    if page == "Home":
        home_page()
    
    elif page == "Settings":
        st.title("Em construção...")


# ====================== EXECUÇÃO DO APLICATIVO STREAMLIT ======================

if __name__ == "__main__":
    OPENAI_API_KEY, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY = get_secrets()
    main()
