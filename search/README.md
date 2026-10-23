# Azure AI Search

Configuração do banco vetorial do PoliGPT. O serviço `srch-poligpt-dev` em si é gerenciado pelo Terraform ([`infra/search.tf`](../infra/search.tf)); os objetos internos (índice, skillset, indexador e fonte de dados) são criados pelos scripts desta pasta, via API do Azure AI Search.

## Fluxo de indexação

```
Azure SQL (poli.tb_poli_knowledge_base)
  │  data source "poligpt-sqldb" (change tracking: só linhas novas/alteradas)
  ▼
Indexador "poligpt-indexer"
  │  skillset "poligpt-skillset"
  │    1. SplitSkill   — divide o documento em trechos de 4000 caracteres (sobreposição de 200)
  │    2. WebApiSkill  — chama a embedding-function para gerar o vetor de cada trecho
  ▼
Índice "poligpt-index" (um documento por trecho, com vetor de 1024 dimensões)
```

O indexador é disparado ao final do pipeline do Databricks (`pipeline/05_indexing/run_search_indexer`).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| [`setup/`](setup/) | Scripts que criam ou atualizam os objetos do AI Search |
| [`embedding-function/`](embedding-function/) | Azure Function chamada pelo skillset para gerar embeddings com a API da OpenAI (`text-embedding-3-large`, 1024 dimensões) |

### Scripts de `setup/`

Execute na ordem abaixo. Todos usam `create_or_update`, então podem ser reexecutados para aplicar mudanças.

| Ordem | Script | Cria |
|---|---|---|
| 1 | `data_source_connector.py` | Fonte de dados `poligpt-sqldb` (Azure SQL) |
| 2 | `create_vector_index.py` | Índice `poligpt-index` (campos, busca vetorial HNSW e configuração semântica) |
| 3 | `create_skillset.py` | Skillset `poligpt-skillset` (chunking + embeddings) |
| 4 | `create_vector_indexer.py` | Indexador `poligpt-indexer`, já disparando uma execução |

## Pré-requisitos

- Python 3.11 com as dependências:
  ```bash
  pip install azure-search-documents==11.5.2 azure-identity azure-keyvault-secrets
  ```
- `az login` com acesso de leitura aos secrets do Key Vault `kv-poligpt-dev-eastus2`.

Os scripts leem endpoints e chaves do Key Vault:

| Secret | Uso |
|---|---|
| `azure-search-endpoint`, `azure-search-admin-key` | Acesso administrativo ao AI Search |
| `sqldb-connection-string` | Conexão da fonte de dados com o Azure SQL |
| `custom-embedding-function-url` | URL da embedding-function chamada pelo skillset (mantido pelo Terraform) |
| `custom-embedding-function-key` | Chave de acesso à embedding-function (mantido pelo Terraform) |

> Se a embedding-function for recriada, o Terraform atualiza os dois últimos secrets; depois é preciso rodar `create_skillset.py` novamente para o skillset apontar para a nova função.

## Limitações do plano gratuito

O plano Free permite 50 MB de armazenamento e 3 índices. O índice atual ocupa praticamente todo o espaço, o que limita a quantidade de documentos indexados.
