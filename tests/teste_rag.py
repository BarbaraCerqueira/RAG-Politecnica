import os
import time

os.environ["AZURESEARCH_FIELDS_ID"] = "chunk_id"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "vector"

from langchain_core.messages import trim_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores.azuresearch import AzureSearch, AzureSearchVectorStoreRetriever

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Conectar ao Azure Key Vault
kv_uri = "https://kv-poligpt-dev-eastus2.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=kv_uri, credential=credential)

OPENAI_API_KEY = client.get_secret("openai-api-key").value
COMPLETION_MODEL = "gpt-4o-2024-08-06"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
MODEL_TEMPERATURE = 0.3

AZURE_SEARCH_ENDPOINT = client.get_secret("azure-search-endpoint").value
AZURE_SEARCH_ADMIN_KEY = client.get_secret("azure-search-admin-key").value
SEARCH_INDEX_NAME = "poligpt-index"
SEARCH_TYPE = "hybrid"
NUM_DOCS_TO_RETRIEVE = 5


def setup_llm(
    temperature: float = 0.3
) -> ChatOpenAI:

    llm = ChatOpenAI(
        model=COMPLETION_MODEL,
        api_key=OPENAI_API_KEY,
        streaming=False,
        temperature=temperature
    )

    return llm

def setup_retriever(
    search_type: str = "hybrid", 
    k: int = 3
) -> AzureSearchVectorStoreRetriever:

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

    memory = MemorySaver()
    llm = setup_llm(temperature=MODEL_TEMPERATURE)
    retriever = setup_retriever(search_type=SEARCH_TYPE, k=NUM_DOCS_TO_RETRIEVE)

    # Definir tool que irá realizar a busca vetorial
    retriever_tool = create_retriever_tool(
        retriever = retriever,
        name = "buscar_poli_info",
        description = "Busca e retorna informações relacionadas a questões acadêmicas ou institucionais sobre a Escola Politécnica da UFRJ.",
    )

    tools = [retriever_tool]

    # Definir mensagem de sistema para direcionar as respostas
    system_message = SystemMessage(
        content="""
        Você é o PoliGPT, um assistente virtual da Escola Politécnica da Universidade Federal do Rio de Janeiro (UFRJ), que auxilia alunos, professores, funcionários e outros interessados 
        em questões acadêmicas e institucionais relacionadas à Escola Politécnica, além de conhecimentos gerais. Quaisquer perguntas relacionadas à área acadêmica feitas pelo
        usuário devem ser por padrão consideradas como referentes à Escola Politécnica, a não ser que seja explicitamente dito o contrário. Caso não tenha informações suficientes 
        para responder à pergunta, diga que não sabe. Responda sempre na mesma língua usada pelo usuário; caso não seja possível reconhecer a língua, use português.
        """
    )

    # Definir agente executor do RAG com acesso às tools propostas
    agent_executor = create_react_agent(
        model=llm, 
        tools=tools, 
        state_modifier=system_message,
        checkpointer=memory
    )

    # Transformar o histórico de mensagens para o formato esperado pelo agente
    processed_history = [system_message]  # A mensagem do sistema deve ser a primeira da lista
    for message in chat_history:
        content = message["content"]
        if message["role"] == "user":
            processed_history += [HumanMessage(content=content)]
        elif message["role"] == "ai":
            processed_history += [AIMessage(content=content)]
        else:
            raise ValueError("O histórico de mensagens contém mensagens inválidas.")
        
    # Adicionar o prompt atual ao histórico
    processed_history += [HumanMessage(content=query)]

    # Remover mensagens antigas do histórico de mensagens
    trimmed_history = trim_messages(
        processed_history,
        strategy="last",
        token_counter=llm,
        max_tokens=8192,  # máximo de tokens que serão "lembrados" pelo modelo
        start_on="human",
        end_on="human",
        include_system=True,
        allow_partial=False
    )

    # Invocar o executor do agente para responder à pergunta do usuário
    config = {"configurable": {"thread_id": session_id}}
    messages = agent_executor.invoke(input={"messages": trimmed_history}, config=config)

    # Extrair resposta final
    output = messages["messages"][-1].content

    return output

if __name__ == '__main__':
    session_id = "001"

    query = "Como me inscrevo numa materia?"

    chat_history = [
        {"role": "user", "content": "Oi! Meu nome é Julia."},
        {"role": "ai", "content": "Ola!"},
        {"role": "user", "content": "Como voce esta?"},
        {"role": "ai", "content": "Bem, obrigado!"},
    ]

    start_time = time.time()
    answer = agent_orchestrator(query, chat_history, session_id)
    end_time = time.time()

    print("Resposta:", answer)
    print("Tempo de execução:", end_time - start_time, "segundos")
