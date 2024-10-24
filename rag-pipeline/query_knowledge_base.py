from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizableTextQuery, 
    VectorizedQuery,
    VectorQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)

kv_uri = f"https://kv-poligpt-dev-eastus2.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=kv_uri, credential=credential)

azure_search_endpoint = client.get_secret("azure-search-endpoint").value
azure_search_admin_key = client.get_secret("azure-search-admin-key").value
azure_search_credential = AzureKeyCredential(azure_search_admin_key)

openai_api_key = client.get_secret("openai-api-key").value
embedding_model = "text-embedding-3-large"
embedding_dimensions = 3072
text_generation_model = "gpt-4"
index_name = "poligpt-index"


# Função que gera embeddings a partir de texto usando a API da OpenAI
def get_embeddings(text, model, dimensions):
    client = OpenAI(api_key = openai_api_key)
    text = text.replace("\n", " ")

    response = client.embeddings.create(
        input=[text],
        model=model,
        dimensions=dimensions
    )

    return response.data[0].embedding


# Hybrid Search
query = "Quais matérias tenho que fazer no quarto período de Engenharia de Computação e Informação?"  
query_embedding = get_embeddings(text=query, model=embedding_model, dimensions=embedding_dimensions)

search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=azure_search_credential)
vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=10, fields="vector", exhaustive=True)
  
results = search_client.search(  
    search_text=query,  
    vector_queries= [vector_query],
    select=["chunk_id", "title", "keywords", "last_modified", "chunk"],
    top=5
)  
  
for result in results:  
    print(f"chunk_id: {result['chunk_id']}")  
    print(f"title: {result['title']}")
    print(f"keywords: {result['keywords']}")  
    print(f"last_modified: {result['last_modified']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"Content: {result['chunk']}")  


######################################

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.llm import LLMChain
from langchain.schema import AIMessage, HumanMessage


# Configuração do GPT-4 no Langchain
llm = ChatOpenAI(model=text_generation_model, temperature=0, api_key=openai_api_key)

# Definir um prompt para GPT-4
template = """Usando os seguintes documentos como contexto, responda à pergunta 
(em caso de não ser suficiente, diga que não sabe):

{query}

Documentos:
{context}
"""

prompt = PromptTemplate(
    input_variables=["query", "context"],
    template=template
)

# Função que envia o contexto (documentos) para o GPT-4 e gera a resposta
def generate_answer(query, documents):
    context = "\n\n".join([doc for doc in documents])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(query=query, context=context)
    
    return response

response = generate_answer(query)
print(response)