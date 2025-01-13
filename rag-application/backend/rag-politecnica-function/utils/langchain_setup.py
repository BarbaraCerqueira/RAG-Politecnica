from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, CosmosDBChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_community.retrievers import AzureAISearchRetriever
from utils import get_secret

def get_chain(conversation_history, text_generation_model):
    # Obtém as credenciais do Key Vault
    openai_api_key = get_secret("openai-api-key")
    openai_api_version = get_secret("OpenAIApiVersion")
    search_service_endpoint = get_secret("AzureSearchServiceEndpoint")
    search_index_name = get_secret("AzureSearchIndexName")

    # Configura o modelo de linguagem
    llm = ChatOpenAI(
        model=text_generation_model,
        api_key=openai_api_key
    )

    # Configura o retriever do Azure Cognitive Search
    retriever = AzureAISearchRetriever(
        service_endpoint=search_service_endpoint,
        index_name=search_index_name,
        api_key=get_secret("AzureSearchApiKey")
    )

    memory = CosmosDBChatMessageHistory(
        
    )

    # Configura a memória da conversa
    memory = ConversationBufferMemory(chat_memory=conversation_history, return_messages=True)

    # Define o template do prompt
    template = """
        Você é um assistente útil. Utilize as informações a seguir para responder à pergunta do usuário.

        Contexto:
        {context}

        Conversa:
        {history}

        Usuário: {input}

        Assistente:
    """

    prompt = PromptTemplate(
        input_variables=["history", "input", "context"],
        template=template
    )

    # Configura o chain do LangChain
    chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        retriever=retriever
    )

    return chain