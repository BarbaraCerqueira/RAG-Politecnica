from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Pegar a OpenAI API Key do Azure Key Vault
def get_openai_api_key():
    key_vault_name = "kv-poligpt-dev-eastus2"
    kv_uri = f"https://{key_vault_name}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=kv_uri, credential=credential)
    secret = client.get_secret("openai-api-key")
    return secret.value

# Função que gera embeddings para uma lista de textos usando a API da OpenAI
def get_embeddings(texts, model, dimensions):
    openai_api_key = get_openai_api_key()

    client = OpenAI(
        api_key = openai_api_key
    )

    texts = [text.replace("\n", " ") for text in texts]

    response = client.embeddings.create(
        input=texts,
        model=model,
        dimensions=dimensions
    )

    embeddings = [item.embedding for item in response.data]
    return embeddings