# Databricks notebook source
# MAGIC %md
# MAGIC # Transformação e Agregação de Dados
# MAGIC Neste Notebook é feita a extração dos dados brutos da camada Raw, seguida da agregação, limpeza e estruturação dos mesmos em uma tabela única. Essa tabela terá como granularidade uma URL (seja de páginas qualquer do site da Politécnica ou de um arquivo PDF), ou seja, cada linha contém o conteúdo de um único arquivo coletado da camada anterior, juntamente com os metadados relevantes para o projeto.  

# COMMAND ----------

import re
import sys
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool

etl_folder_path = './../../etl-pipeline/'
sys.path.append(etl_folder_path)

from delta.tables import DeltaTable
from pyspark.sql import Row
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import col, lit, when, length, udf, to_timestamp, regexp_replace, trim
from utils import list_all_files, upsert_to_delta_lake

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definição de Parâmetros

# COMMAND ----------

input_base_path = "/mnt/adlsraw/pdf/"
output_path = "/mnt/adlstrusted/poli_consolidated_pdf/"
num_processes = 4

# COMMAND ----------

# Widgets para receber os inputs do Workflow
# base_path_pdf = dbutils.widgets.get("base_path_pdf")
# output_path = dbutils.widgets.get("output_path")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funções

# COMMAND ----------

# Função para extrair o conteúdo textual e metadados de um PDF localizado no ADLS
def extract_data_from_pdf(pdf_path):    
    try:
        with fitz.open(f"/dbfs{pdf_path}") as pdf:
            metadata = pdf.metadata if pdf is not None else {}
            text = ""
            for page in pdf:
                text += page.get_text("text")
        return text, metadata
    except Exception as e:
        print(f"Erro ao processar {pdf_path}: {e}")
        return '', {}

# Função para gerar o Row com o conteúdo do PDF
def process_pdf(pdf_path):
    pdf_text, pdf_metadata = extract_data_from_pdf(pdf_path)

    if not isinstance(pdf_metadata, dict):
        pdf_metadata = {}

    # Garantir o retorno como string para evitar incompatibilidade de schema
    def get_metadata_value(key):
        value = pdf_metadata.get(key, 'N/A')
        return str(value) if value is not None else 'N/A'

    return Row(
        file_path=str(pdf_path),
        title=get_metadata_value('title'),
        author=get_metadata_value('author'),
        creator=get_metadata_value('creator'),
        subject=get_metadata_value('subject'),
        creation_date=get_metadata_value('creationDate'),
        modification_date=get_metadata_value('modDate'),
        content=str(pdf_text) if pdf_text is not None else ''
    )

# COMMAND ----------

# Função para analisar e formatar a data extraída do PDF
def parse_pdf_date(pdf_date_str):
    try:
        if pdf_date_str in (None, 'N/A'):
            return None
        
        date_str = pdf_date_str.strip()
        if date_str.startswith('D:'):
            date_str = date_str[2:]
        
        # Regex para extrair os componentes da data
        pattern = r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(Z|[+\-]\d{2}'?(\d{2})'?)?$"
        match = re.match(pattern, date_str)
        
        if not match:
            # Tentar correspondência com apenas data (YYYYMMDD)
            pattern = r"^(\d{4})(\d{2})(\d{2})$"
            match = re.match(pattern, date_str)
            if not match:
                return None
            else:
                year, month, day = match.groups()
                dt_str = f"{year}-{month}-{day} 00:00:00+00:00"
                return dt_str
        else:
            year, month, day, hour, minute, second, tz, tz_minute = match.groups()
            dt_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
            if tz:
                if tz == 'Z':
                    tz_str = '+00:00'
                else:
                    # Remove apóstrofos e adiciona dois pontos entre horas e minutos
                    tz = tz.replace("'", "")
                    if tz_minute:
                        tz_str = f"{tz[:3]}:{tz_minute}"
                    else:
                        tz_str = f"{tz[:3]}:00"
                dt_str += tz_str
            else:
                # Se não tiver fuso horário, assumir UTC
                dt_str += '+00:00'
            return dt_str
    except Exception as e:
        return None

# COMMAND ----------

# Função para verificar a proporção de caracteres válidos num texto
def valid_char_ratio(text):
    if text is None:
        return 0
    total_chars = len(text)
    if total_chars == 0:
        return 0
    valid_chars = len(re.findall(r'[A-Za-z0-9.,!?;: ]', text))
    return valid_chars / total_chars

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extração e Transformação dos Dados 
# MAGIC

# COMMAND ----------

# Multithreading para paralelizar o processo de leitura dos PDFs e HTMLs
pool = ThreadPool(processes=num_processes)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extração e Agregação do Conteúdo dos PDFs

# COMMAND ----------

# Listar todos os PDFs na estrutura de subdiretórios
pdf_files = list_all_files(base_dir=input_base_path, file_format="pdf")

# Processar os PDFs em paralelo
pdf_text_data = pool.map(process_pdf, pdf_files)

# Criar o DataFrame a partir da lista de Rows
df_pdf = spark.createDataFrame(pdf_text_data)

# Fechar o pool de processos
pool.close()
pool.join()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Limpeza e Estruturação dos Dados

# COMMAND ----------

# Registrar função de formatação de data como UDF no Spark
parse_pdf_date_udf = udf(parse_pdf_date, StringType())

# Registrar função para verificar a proporção de caracteres válidos como UDF no Spark
valid_char_ratio_udf = udf(valid_char_ratio, DoubleType())

# COMMAND ----------

# Transformar as datas para o formato Timestamp padrão
df_pdf_processed = (
    df_pdf
    .withColumnRenamed('creation_date', 'creation_date_pdf')
    .withColumnRenamed('modification_date', 'modification_date_pdf')
    .withColumn('parsed_creation_date_str', parse_pdf_date_udf(col('creation_date_pdf')))
    .withColumn('parsed_modification_date_str', parse_pdf_date_udf(col('modification_date_pdf')))
    .withColumn('creation_date', to_timestamp(col('parsed_creation_date_str'), "yyyy-MM-dd HH:mm:ssXXX"))
    .withColumn('modification_date', to_timestamp(col('parsed_modification_date_str'), "yyyy-MM-dd HH:mm:ssXXX"))
)

# COMMAND ----------

df_poli_consolidated_pdf = (
    df_pdf_processed
    .withColumn("content", regexp_replace(col("content"), "(?i)(figura|fig|figure|imagem)\s*\d+[:\-].*", ""))  # remover legendas de figuras
    .withColumn("content", trim(regexp_replace(col("content"), "[ ]+", " ")))  # normalizar espaços no texto
    .withColumn("content", regexp_replace(col("content"), "(\n| \n){2,}", "\n\n"))  # normalizar quebras de linha no texto
    .filter(valid_char_ratio_udf(col("content")) > 0.8)  # desconsiderar arquivos com muitos caracteres especiais
    .filter(length(col("content")) > 200)  # desconsiderar arquivos muito curtos
    .filter(col("content").isNotNull())
    .dropDuplicates(["content"])
    .replace('N/A', None)
    .replace('', None)
    .select(
        col("file_path").cast("string"),
        col("title").cast("string"),
        col("author").cast("string"),
        col("creator").cast("string"),
        col("creation_date").cast("timestamp"),
        col("modification_date").cast("timestamp"),
        col("content").cast("string")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escrita no Data Lake

# COMMAND ----------

upsert_to_delta_lake(df=df_poli_consolidated_pdf, delta_table_path=output_path, key_column="file_path")
