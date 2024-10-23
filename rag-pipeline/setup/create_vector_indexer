from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexer,
    FieldMapping
)

kv_uri = "https://kv-poligpt-dev-eastus2.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=kv_uri, credential=credential)

azure_search_endpoint = client.get_secret("azure-search-endpoint").value
azure_search_admin_key = client.get_secret("azure-search-admin-key").value
azure_search_credential = AzureKeyCredential(azure_search_admin_key)
project_name = "poligpt"


# Create an indexer  
indexer_name = f"{project_name}-indexer"

indexer = SearchIndexer(  
    name=indexer_name,  
    description="Indexer to index documents and generate embeddings",  
    skillset_name=f"{project_name}-skillset",
    target_index_name=f"{project_name}-index",  
    data_source_name=f"{project_name}-sqldb",
    field_mappings=[
        FieldMapping(source_field_name="sk_document", target_field_name="chunk_id")
    ]
)  

indexer_client = SearchIndexerClient(azure_search_endpoint, azure_search_credential)  
indexer_result = indexer_client.create_or_update_indexer(indexer)  
print(f"Indexer created or updated: {indexer.name}")  
  
# Run the indexer  
indexer_client.run_indexer(indexer_name)  
print(f'{indexer.name} is running. If queries return no results, please wait a bit and try again.')  