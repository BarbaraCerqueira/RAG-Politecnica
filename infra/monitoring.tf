# Workspace de logs exigido pelo Application Insights (o modelo "clássico" foi descontinuado).
resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-poligpt-dev"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  # Teto diário de ingestão para não gerar custo (o plano inclui 5 GB/mês gratuitos).
  daily_quota_gb = 0.1
}

resource "azurerm_application_insights" "main" {
  name                = "appi-poligpt-dev"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
}
