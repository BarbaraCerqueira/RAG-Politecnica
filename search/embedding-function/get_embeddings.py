from functools import lru_cache
from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Pegar a OpenAI API Key do Azure Key Vault (lida uma vez por instância e reaproveitada nas próximas requisições)
@lru_cache(maxsize=1)
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

    # max_retries: ao atingir o limite de tokens por minuto (erro 429), o SDK espera o tempo indicado
    # pela OpenAI e tenta de novo, em vez de devolver erro ao indexador do AI Search.
    client = OpenAI(
        api_key = openai_api_key,
        max_retries = 8
    )

    texts = [text.replace("\n", " ") for text in texts]

    response = client.embeddings.create(
        input=texts,
        model=model,
        dimensions=dimensions
    )

    embeddings = [item.embedding for item in response.data]
    return embeddings