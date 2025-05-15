# Databricks notebook source
# MAGIC %md
# MAGIC # Indexação dos Dados
# MAGIC Neste notebook é feita a execução do indexador do Azure Search AI, o qual irá capturar registros novos ou atualizados do Banco SQL, aplicar o pipeline de tratamento definido no serviço (chunking e vetorização) e armazená-los no Search Index para que o RAG possa ter acesso a essas informações.

# COMMAND ----------

import sys
etl_folder_path = './../../etl-pipeline/'
sys.path.append(etl_folder_path)
from utils import get_indexer_client, run_indexer_with_retry

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definição de Parâmetros

# COMMAND ----------

azure_search_endpoint = dbutils.secrets.get("poligpt-secret-scope", "azure-search-endpoint")
azure_search_admin_key = dbutils.secrets.get("poligpt-secret-scope", "azure-search-admin-key")
project_name = "poligpt"

# COMMAND ----------

# # Widgets para receber os inputs do Workflow
# project_name = dbutils.widgets.get("project_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Executar Indexador
# MAGIC

# COMMAND ----------

# Obter cliente do Indexador
indexer_name = f"{project_name}-indexer"
indexer_client = get_indexer_client(indexer_name, azure_search_endpoint, azure_search_admin_key)

# Rodar o indexador
result = run_indexer_with_retry(indexer_client, indexer_name, max_retries=3)
if result == False:
    raise RuntimeError("Indexador falhou após múltiplas tentativas.")
