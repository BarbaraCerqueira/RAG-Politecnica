# Workspace do Databricks. O conteúdo interno (job, clusters, secret scope)
# será gerenciado pelo provider databricks numa etapa seguinte.
resource "azurerm_databricks_workspace" "main" {
  name                = "dbw-poligpt-dev-eastus2"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "standard"

  lifecycle {
    prevent_destroy = true
    # TEMPORÁRIO: atributo sem valor no recurso importado (o padrão já é true).
    # Removido na etapa de migração para o plano premium.
    ignore_changes = [public_network_access_enabled]
  }
}
