resource "azurerm_application_insights" "main" {
  name                = "appi-poloigpt-dev"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  application_type    = "web"
  sampling_percentage = 0
}
