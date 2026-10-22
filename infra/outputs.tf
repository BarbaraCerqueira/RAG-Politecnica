output "chat_api_url" {
  description = "URL base da API do chat."
  value       = "https://${azurerm_linux_function_app.chat.default_hostname}/api"
}

output "frontend_url" {
  description = "URL do site do chat."
  value       = "https://${azurerm_static_web_app.frontend.default_host_name}"
}

output "search_endpoint" {
  value = "https://${azurerm_search_service.main.name}.search.windows.net"
}

output "databricks_workspace_url" {
  value = "https://${azurerm_databricks_workspace.main.workspace_url}"
}

output "key_vault_uri" {
  value = azurerm_key_vault.main.vault_uri
}
