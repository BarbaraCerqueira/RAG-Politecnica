from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SqlIntegratedChangeTrackingPolicy
)

kv_uri = "https://kv-poligpt-dev-eastus2.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=kv_uri, credential=credential)

azure_search_endpoint = client.get_secret("azure-search-endpoint").value
azure_search_admin_key = client.get_secret("azure-search-admin-key").value
azure_search_credential = AzureKeyCredential(azure_search_admin_key)
sqldb_connection_string = client.get_secret("sqldb-connection-string").value
sqldb_table_name = "poli.tb_poli_knowledge_base"
project_name = "poligpt"


# Create a data source 
indexer_client = SearchIndexerClient(azure_search_endpoint, azure_search_credential)
container = SearchIndexerDataContainer(name=sqldb_table_name)
data_source_connection = SearchIndexerDataSourceConnection(
    name=f"{project_name}-sqldb",
    type="azuresql",
    connection_string=sqldb_connection_string,
    container=container,
    data_change_detection_policy=SqlIntegratedChangeTrackingPolicy(),
    data_deletion_detection_policy=None
)
data_source = indexer_client.create_or_update_data_source_connection(data_source_connection)

print(f"Data source created or updated: {data_source.name}")