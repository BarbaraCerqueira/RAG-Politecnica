# Permissões do service principal usado pelos notebooks do Databricks (montagem do Data Lake).
locals {
  pipeline_sp_storage_roles = toset([
    "Storage Blob Data Contributor",
    "Storage Queue Data Contributor",
    "Storage Account Contributor",
  ])
}

resource "azurerm_role_assignment" "pipeline_sp_storage" {
  for_each = local.pipeline_sp_storage_roles

  scope                = azurerm_storage_account.datalake.id
  role_definition_name = each.key
  principal_id         = var.pipeline_sp_object_id
}

resource "azurerm_role_assignment" "pipeline_sp_eventgrid" {
  scope                = azurerm_resource_group.main.id
  role_definition_name = "EventGrid EventSubscription Contributor"
  principal_id         = var.pipeline_sp_object_id
}

resource "azurerm_role_assignment" "pipeline_sp_key_vault" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Contributor"
  principal_id         = var.pipeline_sp_object_id
}
