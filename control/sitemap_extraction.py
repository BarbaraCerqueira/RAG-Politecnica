# Databricks notebook source
# MAGIC %md
# MAGIC # Extração do Sitemap da Politécnica

# COMMAND ----------

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pyspark.sql.functions import col, to_timestamp, cast, current_timestamp

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definição de Parâmetros

# COMMAND ----------

# Definição do User-Agent para identificar o bot
HEADERS = {'User-Agent': 'PoliGPT/1.0'}

# URL do arquivo Sitemap Index contendo os outros sitemaps do site
SITEMAP_INDEX_URL = 'http://poli.ufrj.br/sitemap_index.xml'

# Caminho de destinho dos dados
OUTPUT_PATH = "/mnt/adlscontrol/sitemap_urls/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funções

# COMMAND ----------

def get_sub_sitemaps(sitemap_index_url):
    """
    Extrai as URLs dos sub-sitemaps a partir do sitemap index.

    Args:
        sitemap_index_url (str): URL do sitemap index.

    Returns:
        list: Lista de URLs dos sub-sitemaps.
    """
    try:
        response = requests.get(sitemap_index_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')  # Requer o parser lxml
        sitemap_tags = soup.find_all('sitemap')
        sub_sitemap_urls = [sitemap.find('loc').text for sitemap in sitemap_tags]
        return sub_sitemap_urls
    except requests.RequestException as e:
        print(f"Erro ao acessar o sitemap index {sitemap_index_url}: {e}")
        return []

def extract_sitemap(sitemap_url):
    """
    Extrai todas as URLs e datas de modificação de um sitemap.

    Args:
        sitemap_url (str): URL do sitemap.

    Returns:
        list: Lista de dicionários contendo 'sitemap', 'url' e 'lastmod'.
    """
    sitemap_data = []
    try:
        response = requests.get(sitemap_url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')  # Requer o parser lxml
        url_tags = soup.find_all('url')
        for url_entry in url_tags:
            loc = url_entry.find('loc').text
            lastmod_tag = url_entry.find('lastmod')
            lastmod = lastmod_tag.text if lastmod_tag else None

            sitemap_data.append({
                'sitemap': sitemap_url,
                'url': loc,
                'lastmod': lastmod
            })
    except requests.RequestException as e:
        print(f"Erro ao acessar o sitemap {sitemap_url}: {e}")
    return sitemap_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extração do Conteúdo do Sitemap

# COMMAND ----------

# Acessar o sitemap principal e extrair os sub-sitemaps
sub_sitemap_urls = get_sub_sitemaps(SITEMAP_INDEX_URL)

if not sub_sitemap_urls:
    print("Nenhum sub-sitemap encontrado.")

# Percorrer cada sub-sitemap e extrair as URLs e datas de modificação das páginas do site
all_sitemap_data = []
for sub_sitemap_url in sub_sitemap_urls:
    print(f"Extraindo dados do sub-sitemap: {sub_sitemap_url}")
    sitemap_data = extract_sitemap(sub_sitemap_url)
    all_sitemap_data.extend(sitemap_data)
    time.sleep(1)  # Intervalo entre requisições

if all_sitemap_data:
    df = spark.createDataFrame(all_sitemap_data)
else:
    print("Nenhum dado foi extraído dos sitemaps.")

# COMMAND ----------

df_final = (
    df
    .withColumn("lastmod", to_timestamp("lastmod"))
    .select(
        col("sitemap").cast("string").alias("ds_sub_sitemap"),
        col("url").cast("string").alias("ds_url"),
        col("lastmod").cast("timestamp").alias("dt_last_modification")
    )
    .distinct()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregamento no Data Lake

# COMMAND ----------

try:
    # Salvar os dados no ADLS
    df_final.write.format("parquet").mode("overwrite").save(OUTPUT_PATH)
    print("Dados salvos com sucesso no Data Lake")
except Exception as e:
    print(f"Erro ao salvar os dados no Data Lake: {e}")
