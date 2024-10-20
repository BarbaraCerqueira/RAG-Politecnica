import os
import json
from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import logging

# Pegar a OpenAI API Key do Azure Key Vault
def get_openai_api_key():
    key_vault_name = "kv-poligpt-dev-eastus2"
    kv_uri = f"https://{key_vault_name}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=kv_uri, credential=credential)
    secret = client.get_secret("openai-api-key")
    return secret.value

# Função que gera embeddings usando a API da OpenAI
def get_embeddings(text):
    openai_api_key = get_openai_api_key()

    client = OpenAI(
        api_key = openai_api_key
    )

    text = text.replace("\n", " ")

    embedding = client.embeddings.create(
        input = [text], 
        model = "text-embedding-3-large",
        dimensions = 3  # TEMPORARIO
    ).data[0].embedding

    return embedding