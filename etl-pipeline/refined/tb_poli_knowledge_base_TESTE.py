# Databricks notebook source
# MAGIC %md
# MAGIC # Criação da Base de Conhecimentos para o RAG da Politécnica
# MAGIC Neste Notebook é feita a extração das tabelas consolidadas com os conteúdos dos PDFs e HTMLs extraídos das páginas do site da Politécnica da UFRJ. Em seguida, é realizada uma união desses dados, limpeza de conteúdos que não serão úteis para este caso de uso e transformação numa tabela capaz de alimentar o banco vetorial que será usado para o RAG.

# COMMAND ----------

import sys
import time

from delta.tables import DeltaTable
from pyspark.sql import Window, Row
from pyspark.sql.types import ArrayType, StringType, FloatType
from pyspark.sql.functions import col, lit, when, length, udf, explode, row_number, xxhash64, coalesce, add_months, current_date, regexp_replace, current_timestamp, max as _max, md5

etl_folder_path = './../../etl-pipeline/'
sys.path.append(etl_folder_path)
from utils import list_all_files, upsert_to_delta_lake

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definição de Parâmetros

# COMMAND ----------

input_path_poli_consolidated_html = "/mnt/adlstrusted/poli_consolidated_html/"
input_path_poli_consolidated_pdf = "/mnt/adlstrusted/poli_consolidated_pdf/"
output_path = "/mnt/adlsrefined/tb_poli_knowledge_base_teste/"
table_name = "tb_poli_knowledge_base"
schema_name = "poli"

# COMMAND ----------

# Widgets para receber os inputs do Workflow
# input_base_path = dbutils.widgets.get("input_base_path")
# output_path = dbutils.widgets.get("output_path")

# COMMAND ----------

# Dados de conexão com o banco de dados
jdbcHostname = "sql-poligpt-dev.database.windows.net"
jdbcPort = 1433
jdbcDatabase = "sqldb-poligpt-dev"
jdbcUrl = f"jdbc:sqlserver://{jdbcHostname}:{jdbcPort};database={jdbcDatabase}"

# Setar propriedades de conexão do jdbc
connectionProperties = {
    "user" : dbutils.secrets.get("poligpt-secret-scope", "sql-admin-user"),
    "password" : dbutils.secrets.get("poligpt-secret-scope", "sql-admin-password"),
    "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver",
    "truncate": "true"
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extração e Transformação dos Dados 
# MAGIC

# COMMAND ----------

df_poli_consolidated_html = spark.read.format("delta").load(input_path_poli_consolidated_html)
df_poli_consolidated_pdf = spark.read.format("delta").load(input_path_poli_consolidated_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filtragem e Agregação dos Dados

# COMMAND ----------

df_html = (
    df_poli_consolidated_html
    # Desconsiderar home page - Não contém informações úteis completas
    .filter(col("url") != "https://poli.ufrj.br/")

    # Para páginas de eventos e notícias, manter apenas os publicados há no máximo 1 ano 
    .filter(
        ~(col("file_path").contains("poli_ufrj_br_noticia")) | 
        ((col("file_path").contains("poli_ufrj_br_noticia")) & (col("last_modified") > add_months(current_date(), -12)))
    )
    .filter(
        ~(col("file_path").contains("poli_ufrj_br_evento")) | 
        ((col("file_path").contains("poli_ufrj_br_evento")) & (col("last_modified") > add_months(current_date(), -12)))
    )

    # Filtrar páginas de cronograma de concurso para docentes
    .filter(~((col("content").contains("Concurso Docente")) & (col("content").contains("Cronograma"))))

    # Ajustes estruturais
    .withColumn("last_modified", col("last_modified").cast("date"))
    .withColumn("document_id", md5(col("file_path")))
    .drop("description", "url")
)

# COMMAND ----------

df_pdf = (
    df_poli_consolidated_pdf
    # Desconsiderar arquivos sobre sessões internas da congregação da Politécnica
    .filter(~(col("content").contains("ANEXO À PAUTA DA SESSÃO")))
    .filter(~(col("content").contains("ATA DA SESSÃO ORDINÁRIA")))
    .filter(~(col("content").contains("ATA DA SESSÃO EXTRAORDINÁRIA")))
    .filter(~(col("content").contains("ATA DA SESSÃO SOLENE")))
    .filter(~(col("content").contains("ATA DA SESSÃO ESPECIAL")))
    .filter(~(col("content").contains("ATA — ESCOLA POLITÉCNICA")))
    .filter(~(col("file_path").contains("congregacao-da-politecnica_pautas_")))

    # Desconsiderar arquivos sobre concursos da Politécnica
    .filter(~(col("file_path").contains("poli_ufrj_br_concurso")))

    # Desconsiderar arquivos de formulários e autodeclarações
    .filter(~(col("file_path").rlike("(?i)formulario")))
    .filter(~((col("file_path").rlike("(?i)declaracao")) & col("content").rlike("(?i)declaro")))

    # Ajustes estruturais
    .withColumn("last_modified", coalesce(col("modification_date"), col("creation_date")))
    .withColumn("last_modified", col("last_modified").cast("date"))
    .withColumn("document_id", md5(col("file_path")))
    .drop("modification_date", "creation_date", "author", "creator")
)

# COMMAND ----------

# Juntar os dados numa base de conhecimento única
df_base = df_html.unionByName(df_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ajustes Finais e Estruturação

# COMMAND ----------

df_tb_poli_knowledge_base = (
    df_base
    # Capturar a parte principal do caminho do arquivo (baseado na url da página)
    .withColumn("file_path_end", regexp_replace(col("file_path"), r".*/(pdf|html)/poli_ufrj_br_", ""))
    .withColumn("file_path_main", regexp_replace(col("file_path_end"), r"_/.*\.(pdf|html)", ""))

    # Separar palavras e frases do caminho do arquivo por virgula e guardar como palavras-chave
    .withColumn("keywords", regexp_replace(col("file_path_main"), r"(_|/|\.)", ","))
    .withColumn("keywords", regexp_replace(col("keywords"), r"-", " ")) 

    # Seleção das colunas que serão relevantes para indexar a base de conhecimento 
    .select(
        col("document_id").cast("string").alias("sk_document"),
        col("keywords").cast("string").alias("ds_keywords"),
        col("title").cast("string").alias("ds_title_document"),
        col("last_modified").cast("date").alias("dt_last_modified"),
        col("content").cast("string").alias("ds_content_document")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escrita no Data Lake

# COMMAND ----------

df_tb_poli_knowledge_base.write.mode("overwrite").format("delta").save(output_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escrita no Banco de Dados SQL

# COMMAND ----------

df_tb_poli_knowledge_base.write.jdbc(
    url=jdbcUrl, 
    table=f"{schema_name}.{table_name}",
    mode="overwrite", 
    properties=connectionProperties
)
