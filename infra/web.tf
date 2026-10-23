# Hospedagem do frontend React (frontend/).
resource "azurerm_static_web_app" "frontend" {
  name                = "static-app-poligpt-dev"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
}
