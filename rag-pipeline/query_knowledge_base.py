from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizableTextQuery, 
    RawVectorQuery,
    VectorizedQuery,
    VectorQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)

kv_uri = "https://kv-poligpt-dev-eastus2.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=kv_uri, credential=credential)

azure_search_endpoint = client.get_secret("azure-search-endpoint").value
azure_search_admin_key = client.get_secret("azure-search-admin-key").value
azure_search_credential = AzureKeyCredential(azure_search_admin_key)
index_name = "poligpt-index"


# Hybrid Search
query = "Quais matérias tenho que fazer no quarto período de ECI?"  

search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=azure_search_credential)
vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=10, fields="vector", exhaustive=True)
  
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