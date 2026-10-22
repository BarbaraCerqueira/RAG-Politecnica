resource "azurerm_resource_group" "main" {
  name     = "rg-poligpt-dev-eastus2"
  location = var.location
}
