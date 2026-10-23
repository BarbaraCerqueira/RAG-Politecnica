output "chat_api_url" {
  description = "Endpoint da API do chat."
  value       = "https://${azurerm_function_app_flex_consumption.chat.default_hostname}/api/chatcompletion"
}

output "embedding_function_url" {
  description = "Endpoint da embedding-function chamada pelo skillset do AI Search."
  value       = "https://${azurerm_function_app_flex_consumption.embedding.default_hostname}/api/custom_web_api/openai_embeddings"
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
