import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definição do User-Agent para identificar o bot
HEADERS = {'User-Agent': 'PoliGPT/1.0'}

# URL do arquivo Sitemap Index contendo os outros sitemaps do site
SITEMAP_INDEX_URL = 'http://poli.ufrj.br/sitemap_index.xml'


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
        logging.error(f"Erro ao acessar o sitemap index {sitemap_index_url}: {e}")
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
        logging.error(f"Erro ao acessar o sitemap {sitemap_url}: {e}")
    return sitemap_data

def main():
    """
    Função principal que coordena a extração dos sitemaps e salva os dados em um arquivo CSV.
    """
    # Acessar o sitemap principal e extrair os sub-sitemaps
    sub_sitemap_urls = get_sub_sitemaps(SITEMAP_INDEX_URL)

    if not sub_sitemap_urls:
        logging.warning("Nenhum sub-sitemap encontrado. Encerrando.")
        return

    # Percorrer cada sub-sitemap e extrair as URLs e datas de modificação
    all_sitemap_data = []
    for sub_sitemap_url in sub_sitemap_urls:
        logging.info(f"Extraindo dados do sub-sitemap: {sub_sitemap_url}")
        sitemap_data = extract_sitemap(sub_sitemap_url)
        all_sitemap_data.extend(sitemap_data)
        time.sleep(1)  # Manter um intervalo entre requisições

    if all_sitemap_data:
        # Salvar os dados em um arquivo CSV
        df = pd.DataFrame(all_sitemap_data)
        df.to_csv('extracted_files/sitemap_urls.csv', index=False)
        logging.info("Dados salvos em 'sitemap_urls.csv'")
    else:
        logging.warning("Nenhum dado foi extraído dos sitemaps.")

if __name__ == '__main__':
    main()
