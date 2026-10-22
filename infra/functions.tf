# ---------------------------------------------------------------------------
# API do chat (backend/)
# ---------------------------------------------------------------------------
resource "azurerm_service_plan" "chat" {
  name                = "ASP-rgpoligptdeveastus2-89e9"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  os_type             = "Linux"
  sku_name            = "Y1"
}

resource "azurerm_linux_function_app" "chat" {
  name                       = "rag-politecnica-function"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = var.location
  service_plan_id            = azurerm_service_plan.chat.id
  storage_account_name       = azurerm_storage_account.datalake.name
  storage_account_access_key = azurerm_storage_account.datalake.primary_access_key
  https_only                 = true
  builtin_logging_enabled    = false
  client_certificate_mode    = "Required"

  app_settings = {
    AZURE_SEARCH_ENDPOINT             = "https://${azurerm_search_service.main.name}.search.windows.net"
    AZURESEARCH_FIELDS_CONTENT        = "chunk"
    AZURESEARCH_FIELDS_CONTENT_VECTOR = "vector"
    AZURESEARCH_FIELDS_ID             = "chunk_id"
    COMPLETION_MODEL                  = "gpt-4o-2024-08-06"
    EMBEDDING_DIMENSIONS              = "1024"
    EMBEDDING_MODEL                   = "text-embedding-3-large"
    MODEL_TEMPERATURE                 = "0"
    NUM_DOCS_TO_RETRIEVE              = "10"
    SEARCH_INDEX_NAME                 = "poligpt-index"
    SEARCH_TYPE                       = "hybrid"
  }

  site_config {
    ftps_state                             = "FtpsOnly"
    application_insights_connection_string = azurerm_application_insights.main.connection_string

    application_stack {
      python_version = "3.11"
    }

    cors {
      allowed_origins = var.chat_cors_allowed_origins
    }
  }

  lifecycle {
    ignore_changes = [
      # TEMPORÁRIO: as chaves ainda estão em texto puro nas configurações da Function.
      # Serão trocadas por referências ao Key Vault na etapa de segurança do backend.
      app_settings["OPENAI_API_KEY"],
      app_settings["AZURE_SEARCH_ADMIN_KEY"],
      # Tags "hidden-link" criadas automaticamente pelo portal para o Application Insights.
      tags,
      # Vazio no recurso importado, equivalente a "Allow".
      site_config[0].ip_restriction_default_action,
      site_config[0].scm_ip_restriction_default_action,
    ]
  }
}

# ---------------------------------------------------------------------------
# Função de embeddings chamada pelo skillset do AI Search (search/embedding-function/)
# ---------------------------------------------------------------------------
resource "azurerm_service_plan" "embedding" {
  name                = "ASP-rgpoligptdeveastus2-9b42"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  os_type             = "Linux"
  sku_name            = "Y1"
}

resource "azurerm_linux_function_app" "embedding" {
  name                                           = "openai-embeddings-function"
  resource_group_name                            = azurerm_resource_group.main.name
  location                                       = var.location
  service_plan_id                                = azurerm_service_plan.embedding.id
  storage_account_name                           = azurerm_storage_account.datalake.name
  storage_account_access_key                     = azurerm_storage_account.datalake.primary_access_key
  https_only                                     = true
  builtin_logging_enabled                        = false
  client_certificate_mode                        = "Required"
  ftp_publish_basic_authentication_enabled       = false
  webdeploy_publish_basic_authentication_enabled = false

  # Identidade usada para ler a chave da OpenAI no Key Vault.
  identity {
    type = "SystemAssigned"
  }

  site_config {
    ftps_state                             = "FtpsOnly"
    application_insights_connection_string = azurerm_application_insights.main.connection_string
    application_insights_key               = azurerm_application_insights.main.instrumentation_key

    application_stack {
      python_version = "3.11"
    }

    cors {
      allowed_origins = ["https://portal.azure.com"]
    }
  }

  lifecycle {
    ignore_changes = [
      # Tags "hidden-link" e sticky settings criadas automaticamente pelo portal para o Application Insights.
      tags,
      sticky_settings,
      # Vazio no recurso importado, equivalente a "Allow".
      site_config[0].ip_restriction_default_action,
      site_config[0].scm_ip_restriction_default_action,
    ]
  }
}
