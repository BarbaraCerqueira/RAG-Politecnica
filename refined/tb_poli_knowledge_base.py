# Databricks notebook source
# MAGIC %md
# MAGIC # Criação da Base de Conhecimentos para o RAG da Politécnica
# MAGIC Neste Notebook é feita a extração das tabelas consolidadas com os conteúdos dos PDFs e HTMLs extraídos das páginas do site da Politécnica da UFRJ. Em seguida, é realizada uma união desses dados, limpeza de conteúdos que não serão úteis para este caso de uso e transformação numa tabela capaz de alimentar o banco vetorial que será usado para o RAG.

# COMMAND ----------

import time
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from delta.tables import DeltaTable
from pyspark.sql import Window, Row
from pyspark.sql.types import ArrayType, StringType, FloatType
from pyspark.sql.functions import col, lit, when, length, udf, explode, row_number, xxhash64, coalesce, add_months, current_date, regexp_replace, current_timestamp, max as _max

from utils import list_all_files, upsert_to_delta_lake

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definição de Parâmetros

# COMMAND ----------

input_path_poli_consolidated_html = "/mnt/adlstrusted/poli_consolidated_html/"
input_path_poli_consolidated_pdf = "/mnt/adlstrusted/poli_consolidated_pdf/"
output_path = "/mnt/adlsrefined/tb_poli_knowledge_base/"
embedding_model = "text-embedding-3-large"
num_embedding_dimensions = 3072
embedding_api_rate_limit = 3000
chunk_size = 2000
chunk_overlap = 200

# COMMAND ----------

# Obter credenciais de API da OpenAI
openai_api_key=dbutils.secrets.get("poligpt-secret-scope", "openai-api-key")
openai_organization_id=dbutils.secrets.get("poligpt-secret-scope", "openai-organization-id")

# COMMAND ----------

# Obter número total de cores disponíveis no cluster
total_cores = int(spark.sparkContext.defaultParallelism)

# COMMAND ----------

# Widgets para receber os inputs do Workflow
# input_base_path = dbutils.widgets.get("input_base_path")
# output_path = dbutils.widgets.get("output_path")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funções

# COMMAND ----------

# Função para dividir o texto em chunks
def split_text(content):
    
    # Configurar o Text Splitter com os delimitadores e tamanhos dos chunks
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    # Dividir o texto em pedaços menores
    if content:
        chunks = text_splitter.split_text(content)
        return chunks
    else:
        return []

# COMMAND ----------

# Função para calcular o tempo recomendado de espera entre requisições de uma API por tarefa simultânea do Spark
# respeitando o limite de taxa de requisições da API (por minuto) e assumindo um aproveitamento completo do paralelismo disponível no cluster
def get_api_request_delay(api_rate_limit):
    api_rate_limit_per_sec = api_rate_limit / 60
    rate_limit_per_core = api_rate_limit_per_sec / total_cores
    return 1.0 / rate_limit_per_core

# COMMAND ----------

# Função para gerar embeddings a partir de texto
def get_embeddings(text):

    # Configurar o modelo de embedding a ser usado
    embeddings_model = OpenAIEmbeddings(
    model=embedding_model,
    dimensions=num_embedding_dimensions,
    api_key=openai_api_key,
    organization=openai_organization_id
    )
 
    try:
        # Gerar embeddings a partir da coluna de texto
        embedding = embeddings_model.embed_query(text)
    except Exception as e:
        embedding = [0.0] * num_embedding_dimensions

    # Espera para respeitar o limite de taxa da API
    request_wait_time = get_api_request_delay(embedding_api_rate_limit)
    time.sleep(request_wait_time)

    return embedding 

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
    .drop("modification_date", "creation_date", "author", "creator")
)

# COMMAND ----------

# Juntar os dados numa base de conhecimento única
df_base = df_html.unionByName(df_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunking dos Textos

# COMMAND ----------

# Registrar a função de chunking como UDF do Spark
split_text_udf = udf(split_text, ArrayType(StringType()))

# COMMAND ----------

# Adicionar coluna com a lista dos chunks obtidos a partir do conteúdo textual
df_base_chunks = df_base.withColumn("chunks", split_text_udf(col("content")))

# Explodir os chunks em linhas separadas
df_base_chunks = df_base_chunks.select(
    col("file_path"),
    col("title"),
    col("last_modified"),
    explode(col("chunks")).alias("chunk")
)

# Adicionar um identificador único do chunk
window = Window.partitionBy("file_path").orderBy("chunk")
df_base_chunks = (
    df_base_chunks
    .withColumn("chunk_index", row_number().over(window))
    .withColumn("id_chunk", xxhash64(col("file_path"), col("chunk_index")))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Geração de Embeddings

# COMMAND ----------

# Reparticionar o DataFrame conforme o número total de núcleos disponíveis
df_repartitioned = df_base_chunks.coalesce(total_cores)

# COMMAND ----------

# Registrar a função de gerar embeddings como UDF do Spark
get_embeddings_udf = udf(get_embeddings, ArrayType(FloatType()))

# Obter embeddings a partir dos chunks
df_embeddings = (
    df_repartitioned
    .withColumn("embedding", get_embeddings_udf(col("chunk")))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ajustes Finais e Estruturação

# COMMAND ----------

df_tb_poli_knowledge_base = (
    df_embeddings
    .withColumn("file_path_short", regexp_replace(col("file_path"), r".*/(pdf|html)/", ""))
    .withColumn("_insert_timestamp", current_timestamp())
    .select(
        col("id_chunk").cast("long").alias("Id_Chunk"),
        col("file_path_short").cast("string").alias("Caminho_Documento"),
        col("title").cast("string").alias("Titulo_Documento"),
        col("last_modified").cast("date").alias("Data_Publicacao"),
        col("chunk").cast("string").alias("Conteudo_Documento"),
        col("embedding").cast("array<float>").alias("Embedding"),
        col("_insert_timestamp").cast("timestamp")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escrita no Data Lake

# COMMAND ----------

upsert_to_delta_lake(df=df_tb_poli_knowledge_base, delta_table_path=output_path, key_column="Id_Chunk")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remover Dados Obsoletos

# COMMAND ----------

# Carregar a tabela Delta com os dados recém inseridos e atualizados como DeltaTable
delta_table = DeltaTable.forPath(spark, output_path)

# Selecionar o timestamp mais recente de inserção dos chunks para cada Documento
latest_timestamps = (
    delta_table.toDF()
    .groupBy("Caminho_Documento")
    .agg(_max("_insert_timestamp").alias("latest_timestamp"))
)

# Associar de volta ao Dataframe original para identificar quais chunks são obsoletos
chunks_to_delete = (
    delta_table.toDF()
    .join(latest_timestamps, on="Caminho_Documento", how="left")
    .filter(col("_insert_timestamp") < col("latest_timestamp"))
    .select("Id_Chunk")
    .collect()
)

if chunks_to_delete:
    print("Iniciando deleção de chunks que se tornaram obsoletos.")
    for row in chunks_to_delete:
        print(f"Deletando Chunk ID: {row["Id_Chunk"]}")
        delta_table.delete(f"Id_Chunk = '{row['Id_Chunk']}'")
else:
    print("Não foram encontrados chunks antigos e obsoletos para deletar.")
