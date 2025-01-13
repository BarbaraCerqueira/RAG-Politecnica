import os
import time
import uuid
import streamlit as st
from pathlib import Path

# Importações do Azure e LangChain
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from langchain_core.messages import trim_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores.azuresearch import AzureSearch, AzureSearchVectorStoreRetriever

#################################
# CONFIGURAÇÕES E VARIÁVEIS GLOBAIS
#################################

# Ajuste o título, layout e ícone da aba do navegador
st.set_page_config(
    page_title="PoliGPT",
    page_icon="imagens/poligpt_icon.ico",
    layout="centered"
)

# Gerar id da sessão
st.session_state.id = uuid.uuid4()

# Ajuste se quiser trabalhar com variáveis de ambiente
os.environ["AZURESEARCH_FIELDS_ID"] = "chunk_id"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "vector"

# Ajuste o caminho para o logo
LOGO_PATH = "images/poligpt_logo.png"

# Substitua pela sua URL do Key Vault
kv_uri = "https://kv-poligpt-dev-eastus2.vault.azure.net"

# Inicializa credenciais do Azure Key Vault
credential = DefaultAzureCredential()
client = SecretClient(vault_url=kv_uri, credential=credential)

# Busque suas chaves/segredos a partir do Key Vault
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

#################################
# FUNÇÕES AUXILIARES
#################################

@st.cache_resource
def setup_llm(
    temperature: float = MODEL_TEMPERATURE
) -> ChatOpenAI:
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
def setup_retriever(
    search_type: str = "hybrid", 
    k: int = 3
) -> AzureSearchVectorStoreRetriever:
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

def agent_orchestrator(
    query: str, 
    chat_history: list,
    session_id: str
) -> str:
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
        retriever = retriever,
        name = "buscar_poli_info",
        description = "Busca e retorna informações relacionadas à Escola Politécnica da UFRJ."
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

    # Formata o histórico de mensagens para o agente
    processed_history = [system_message]  # A mensagem do sistema vai na frente
    for message in chat_history:
        content = message["content"]
        if message["role"] == "user":
            processed_history.append(HumanMessage(content=content))
        elif message["role"] == "ai":
            processed_history.append(AIMessage(content=content))
        else:
            # Mensagens inválidas ou outros roles podem ser tratados de outra forma
            raise ValueError("O histórico de mensagens contém mensagens inválidas.")

    # Adiciona a última pergunta do usuário
    processed_history.append(HumanMessage(content=query))

    # Remove mensagens antigas (se necessário) para não ultrapassar limite de tokens
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

    # Invoca o agente com o histórico ajustado
    config = {"configurable": {"thread_id": session_id}}
    messages = agent_executor.invoke(input={"messages": trimmed_history}, config=config)

    # Extrai a resposta final do modelo
    output = messages["messages"][-1].content
    return output

#################################
# FUNÇÃO PRINCIPAL DA INTERFACE
#################################

def main():
    # CSS customizado para melhorar o layout/estilo
    custom_css = """
    <style>
    /* Centraliza o container principal */
    .main > div {
        max-width: 800px;
        margin: 0 auto;
    }
    /* Estilo do histórico de mensagens */
    .message {
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #DCF8C6;
        text-align: left;
    }
    .ai-message {
        background-color: #E5E5EA;
        text-align: left;
    }
    /* Campo de texto personalizado */
    div[data-baseweb="input"] {
        margin-top: 15px;
    }
    /* Título da página */
    h1 {
        text-align: center;
        margin-bottom: 5px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Exibe o logo no topo
    if Path(LOGO_PATH).is_file():
        st.image(LOGO_PATH, use_container_width=False)
    else:
        # Caso não encontre o arquivo, use um texto
        st.markdown(f"PoliGPT")

    # st.title("PoliGPT - Chatbot RAG")
    st.markdown(
        """
        
        ## PoliGPT - Seu assistente virtual acadêmico 
        """,
        unsafe_allow_html=True
    )

    # Inicialização do histórico de chat na sessão
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Container para exibir o histórico de mensagens
    with st.container():
        for idx, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(
                    f'<div class="message user-message"><strong>Você:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            elif message["role"] == "ai":
                st.markdown(
                    f'<div class="message ai-message"><strong>PoliGPT:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )

    # Campo para digitar uma mensagem ao RAG
    prompt = st.chat_input("Escreva uma mensagem...")

    if prompt:
        # Armazena a mensagem do usuário
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Gera a resposta do modelo
        session_id = st.session_state.id
        start_time = time.time()
        answer = agent_orchestrator(prompt, st.session_state.chat_history, session_id)
        end_time = time.time()

        # Armazena a resposta do PoliGPT
        st.session_state.chat_history.append({"role": "ai", "content": answer})

        # Apresenta dados de performance (opcional)
        execution_time = end_time - start_time
        st.write(f"*(Tempo de execução: {execution_time:.2f} segundos)*")

        # Força a atualização do layout
        st.rerun()

#################################
# EXECUÇÃO DO APLICATIVO STREAMLIT
#################################

if __name__ == "__main__":
    main()
