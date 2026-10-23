# Data Lake (ADLS Gen2) com as camadas do pipeline.
resource "azurerm_storage_account" "datalake" {
  name                            = "dlspoligptdev"
  resource_group_name             = azurerm_resource_group.main.name
  location                        = var.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  is_hns_enabled                  = true
  allow_nested_items_to_be_public = false

  lifecycle {
    prevent_destroy = true
  }
}

# Camadas do pipeline de dados (os containers internos das Functions são gerenciados pela própria Azure).
resource "azurerm_storage_container" "layers" {
  for_each = toset(["control", "raw", "trusted", "refined"])

  name               = each.key
  storage_account_id = azurerm_storage_account.datalake.id

  lifecycle {
    prevent_destroy = true
  }
}

# Storage interno das Function Apps (estado do runtime e pacotes de deploy), separado dos dados do pipeline.
resource "azurerm_storage_account" "functions" {
  name                            = "stpoligptfuncdev"
  resource_group_name             = azurerm_resource_group.main.name
  location                        = var.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
}

# Containers onde o plano Flex Consumption guarda o pacote de código de cada Function.
resource "azurerm_storage_container" "function_deployments" {
  for_each = toset(["chat", "embedding"])

  name                  = "deploy-${each.key}"
  storage_account_id    = azurerm_storage_account.functions.id
  container_access_type = "private"
}
