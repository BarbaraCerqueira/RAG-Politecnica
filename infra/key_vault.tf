# Key Vault com os segredos do projeto. Os valores dos secrets são gerenciados fora do Terraform
# (az keyvault secret set), para que nunca apareçam no código nem no state.
resource "azurerm_key_vault" "main" {
  name                = "kv-poligpt-dev-eastus2"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  tenant_id           = var.tenant_id
  sku_name            = "standard"

  lifecycle {
    prevent_destroy = true
  }
}

resource "azurerm_key_vault_access_policy" "admin" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = var.tenant_id
  object_id    = var.admin_object_id

  secret_permissions = ["Get", "List", "Set", "Delete", "Recover", "Backup", "Restore"]
}

# Necessário para o secret scope do Databricks baseado em Key Vault (poligpt-secret-scope).
resource "azurerm_key_vault_access_policy" "databricks" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = var.tenant_id
  object_id    = var.databricks_app_object_id

  secret_permissions = ["Get", "List"]
}

resource "azurerm_key_vault_access_policy" "embedding_function" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = var.tenant_id
  object_id    = azurerm_linux_function_app.embedding.identity[0].principal_id

  secret_permissions = ["Get", "List", "Set", "Delete", "Recover", "Backup", "Restore"]
}
