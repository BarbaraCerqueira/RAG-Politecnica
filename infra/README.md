# Infraestrutura (Terraform)

Infraestrutura do PoliGPT na Azure como código. Todos os recursos do resource group `rg-poligpt-dev-eastus2` foram importados a partir do ambiente existente.

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
| `storage.tf` | Data Lake (ADLS Gen2) e containers das camadas |
| `key_vault.tf` | Key Vault e políticas de acesso |
| `sql.tf` | Servidor e banco Azure SQL |
| `search.tf` | Azure AI Search |
| `functions.tf` | Function Apps do chat e da função de embeddings |
| `web.tf` | Static Web App do frontend |
| `monitoring.tf` | Application Insights |
| `databricks.tf` | Workspace do Databricks, secret scope (Key Vault), Git folder, cluster interativo e job do pipeline |
| `iam.tf` | Permissões (role assignments) |

## State remoto

O state fica em `rg-poligpt-tfstate/stpoligpttfstate/tfstate/poligpt-dev.tfstate`, fora do resource group gerenciado. O storage account tem acesso por chave desativado (só Entra ID), versionamento, soft delete de 30 dias e um lock contra exclusão. Foi criado manualmente, fora do Terraform.

## Segredos

Nenhum segredo fica no código ou nos `.tfvars`. Os valores são guardados no Key Vault (`kv-poligpt-dev-eastus2`) e lidos em tempo de execução. A senha do SQL é lida como recurso efêmero e enviada como atributo *write-only*, sem ser gravada no state.

## Proteções

Recursos que guardam dados (Data Lake, Key Vault, SQL, AI Search, Databricks) têm `prevent_destroy = true`: o Terraform se recusa a destruí-los.
