# Databricks notebook source
# MAGIC %md
# MAGIC # Transformação e Agregação de Dados
# MAGIC Neste Notebook é feita a extração dos dados brutos da camada Raw, seguida da agregação, limpeza e estruturação dos mesmos em uma tabela única. Essa tabela terá como granularidade uma URL (seja de páginas qualquer do site da Politécnica ou de um arquivo PDF), ou seja, cada linha contém o conteúdo de um único arquivo coletado da camada anterior, juntamente com os metadados relevantes para o projeto.  

# COMMAND ----------

import sys
from bs4 import BeautifulSoup
from multiprocessing.pool import ThreadPool

etl_folder_path = './../../etl-pipeline/'
sys.path.append(etl_folder_path)

from delta.tables import DeltaTable
from pyspark.sql import Row
from pyspark.sql.functions import col, lit, when, length, trim, regexp_replace, unix_timestamp, current_timestamp
from utils import list_all_files, upsert_to_delta_lake, save_checkpoint, get_checkpoint_by_source

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definição de Parâmetros

# COMMAND ----------

input_base_path = "/mnt/adlsraw/html/"
output_path = "/mnt/adlstrusted/poli_consolidated_html/"
checkpoint_path = "/mnt/adlstrusted/checkpoints/poli_consolidated_html/"
num_processes = 4

# COMMAND ----------

# Widgets para receber os inputs do Workflow
# input_base_path = dbutils.widgets.get("input_base_path")
# output_path = dbutils.widgets.get("output_path")
# checkpoint_path = dbutils.widgets.get("checkpoint_path")
# num_processes = dbutils.widgets.get("num_processes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funções

# COMMAND ----------

def extract_relevant_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Tentar encontrar a divisão com o conteúdo principal da página
    main_div = soup.find('main', id='primary')

    # Elementos indesejados dentro do conteúdo principal
    unwanted_selectors = [
        {'name': 'div', 'attrs': {'class': 'poli-post'}},                         # Compartilhamento de post
        {'name': 'div', 'attrs': {'class': 'breadcrumb-poli'}},                   # Caminho da seção
        {'name': 'div', 'attrs': {'class': 'block-linkexterno linkexterno'}},     # Links externos
        {'name': 'div', 'attrs': {'class': 'block-linkarquivo linkarquivo'}},     # Links para arquivos
        {'name': 'a', 'attrs': {'class': 'wp-block-social-link-anchor'}},         # Links para redes sociais
        {'name': 'nav', 'attrs': {'class': 'navigation pagination'}}              # Páginas de navegação
    ]
    
    for selector in unwanted_selectors:
        for tag in main_div.find_all(selector['name'], attrs=selector['attrs']):
            tag.decompose()

    # Extrair o texto limpo
    text = main_div.get_text(separator='\n', strip=True)
    
    return text

def extract_metadata(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    title = soup.find("meta", attrs={"property": "og:title"})
    title = title['content'] if title else 'N/A'

    description = soup.find("meta", attrs={"property": "og:description"})
    description = description['content'] if description else 'N/A'

    url = soup.find("meta", attrs={"property": "og:url"})
    url = url['content'] if url else 'N/A'

    modified_time = soup.find("meta", attrs={"property": "article:modified_time"})
    modified_time = modified_time['content'] if modified_time else 'N/A'

    page_metadata = {
        'URL': url,
        'Title': title,
        'Description': description,
        'Last Modified': modified_time
    }

    return page_metadata

# COMMAND ----------

# Função para ler o conteúdo completo de um HTML localizado no ADLS
def extract_text_from_html(html_path):
    try:
        with open(f"/dbfs{html_path}", "r", encoding="utf-8") as html_file:
            return html_file.read()
    except Exception as e:
        print(f"Erro ao processar {html_path}: {e}")
        return None
    
# Função para gerar o Row com o conteúdo do HTML
def process_html(html_path):
    html_content = extract_text_from_html(html_path)
    metadata = extract_metadata(html_content)
    relevant_content = extract_relevant_content(html_content)
    return Row(file_path=html_path, 
               url=metadata['URL'], 
               title=metadata['Title'],
               description=metadata['Description'], 
               last_modified=metadata['Last Modified'], 
               content=relevant_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extração e Transformação dos Dados 
# MAGIC

# COMMAND ----------

# Encontrar timestamp da última extração para evitar reprocessamento
last_extraction = get_checkpoint_by_source(checkpoint_path=checkpoint_path, source_path=input_base_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extração e Agregação do Conteúdo dos HTMLs

# COMMAND ----------

# Listar todos os HTMLs na estrutura de subdiretórios desde a última extração
html_files = list_all_files(base_dir=input_base_path, file_format="html", min_modified_timestamp=last_extraction)

# COMMAND ----------

if html_files:
    # Multithreading para paralelizar o processo de leitura de PDFs ou HTMLs
    pool = ThreadPool(processes=num_processes)

    # Processar os HTMLs em paralelo
    html_text_data = pool.map(process_html, html_files)

    # Criar o DataFrame a partir da lista de Rows
    df_html = spark.createDataFrame(html_text_data)

    # Fechar o pool de processos
    pool.close()
    pool.join()

else:
    dbutils.notebook.exit("Não foram encontrados arquivos HTML novos ou atualizados para processar. O processamento deste notebook será interrompido.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Limpeza e Estruturação dos Dados

# COMMAND ----------

df_poli_consolidated_html = (
    df_html
    .withColumn("content", trim(regexp_replace(col("content"), "[ ]+", " ")))  # normalizar espaços no texto
    .filter(~col("content").contains("Lorem ipsum"))  # desconsiderar páginas de teste sem conteúdo relevante
    .filter(~col("content").rlike("Nothing Found"))  # desconsiderar páginas com erro
    .filter(length(col("content")) > 100)  # desconsiderar páginas com muito pouco conteúdo
    .filter(col("content").isNotNull())
    .dropDuplicates(["content"])
    .replace('N/A', None)
    .select(
        col("file_path").cast("string"),
        col("url").cast("string"),
        col("title").cast("string"),
        col("last_modified").cast("timestamp"),
        col("description").cast("string"),
        col("content").cast("string"),
        # Adicionando data de atualização do registro em milissegundos para usar nas leituras incrementais
        (unix_timestamp(current_timestamp()) * 1000).cast("long").alias("_update_timestamp")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escrita no Data Lake

# COMMAND ----------

upsert_to_delta_lake(df=df_poli_consolidated_html, delta_table_path=output_path, key_column="file_path")
save_checkpoint(checkpoint_path=checkpoint_path, source_path=input_base_path, target_path=output_path)
