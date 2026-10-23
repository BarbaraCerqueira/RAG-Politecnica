# Infraestrutura (Terraform)

Infraestrutura do PoliGPT na Azure como código. Os recursos do resource group `rg-poligpt-dev-eastus2` foram importados a partir do ambiente existente, com alguns pequenos ajustes posteriores.

## Pré-requisitos

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.6
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) autenticada: `az login --tenant <tenant_id>`
- Permissão **Storage Blob Data Contributor** no storage account do state (`stpoligpttfstate`)

## Uso

```bash
cd infra
terraform init      # conecta ao state remoto e baixa os providers
terraform plan      # mostra o que mudaria — sempre revisar antes de aplicar
terraform apply
```

## Organização

| Arquivo | Recursos |
|---|---|
| `versions.tf` | Versões, backend do state e configuração do provider |
| `main.tf` | Resource group |
| `storage.tf` | Data Lake (ADLS Gen2) com os containers das camadas, e storage interno das Function Apps |
| `key_vault.tf` | Key Vault, políticas de acesso e os secrets derivados de outros recursos (URL e chave da embedding-function) |
| `sql.tf` | Servidor e banco Azure SQL |
| `search.tf` | Serviço do Azure AI Search (os objetos internos são criados por `search/setup/`) |
| `functions.tf` | Function Apps do chat e da embedding-function, no plano Flex Consumption |
| `web.tf` | Static Web App do frontend |
| `monitoring.tf` | Log Analytics e Application Insights |
| `databricks.tf` | Workspace do Databricks, secret scope (Key Vault), Git folder, cluster interativo e job do pipeline |
| `iam.tf` | Permissões (role assignments) |

## State remoto

O state fica em `rg-poligpt-tfstate/stpoligpttfstate/tfstate/poligpt-dev.tfstate`, fora do resource group gerenciado. O storage account tem acesso por chave desativado (só Entra ID), versionamento, soft delete de 30 dias e um lock contra exclusão. Foi criado manualmente, fora do Terraform.

## Segredos

Nenhum segredo fica no código ou nos `.tfvars`. Os valores são guardados no Key Vault (`kv-poligpt-dev-eastus2`) e lidos em tempo de execução:

- A Function do chat lê a chave da OpenAI e do AI Search por **referências ao Key Vault** nas app settings (`@Microsoft.KeyVault(...)`), usando sua identidade gerenciada.
- A embedding-function lê a chave da OpenAI diretamente do Key Vault no código, também com identidade gerenciada.
- A senha do SQL é lida como recurso efêmero e enviada como atributo *write-only*, sem ser gravada no state.

## Deploy do código das Functions

O Terraform cria as Function Apps, mas não publica o código. O deploy é feito com o Azure Functions Core Tools:

```bash
cd backend && func azure functionapp publish func-poligpt-chat-dev --python
cd search/embedding-function && func azure functionapp publish func-poligpt-embed-dev --python
```

Se a embedding-function for recriada, rode `search/setup/create_skillset.py` para o skillset do AI Search apontar para a nova URL e chave.

## Proteções

Recursos que guardam dados (Data Lake, Key Vault, SQL, AI Search, Databricks) têm `prevent_destroy = true`: o Terraform se recusa a destruí-los.
