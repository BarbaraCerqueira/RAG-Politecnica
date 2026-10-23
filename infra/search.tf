# Banco vetorial. O índice, skillset, indexer e data source (plano de dados) são criados
# pelos scripts em search/setup, não pelo Terraform (ver search/README.md).
resource "azurerm_search_service" "main" {
  name                = "srch-poligpt-dev"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "free"

  lifecycle {
    prevent_destroy = true
  }
}
