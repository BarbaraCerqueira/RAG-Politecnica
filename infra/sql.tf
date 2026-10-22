# Senha do admin do SQL, lida do Key Vault só durante a execução (efêmera: não vai para o state).
ephemeral "azurerm_key_vault_secret" "sql_admin_password" {
  name         = "sql-admin-password"
  key_vault_id = azurerm_key_vault.main.id
}

# Banco SQL que recebe a base de conhecimento (camada refined) e serve de fonte para o indexador do AI Search.
resource "azurerm_mssql_server" "main" {
  name                = "sql-poligpt-dev"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  version             = "12.0"
  administrator_login = "sql-poligpt-admin"

  # Write-only: o valor é enviado à Azure mas nunca gravado no state.
  # Para trocar a senha: atualizar o secret no Key Vault e incrementar a versão.
  administrator_login_password_wo         = ephemeral.azurerm_key_vault_secret.sql_admin_password.value
  administrator_login_password_wo_version = 1

  lifecycle {
    prevent_destroy = true
    # TEMPORÁRIO: evita reenviar a senha durante a importação inicial.
    ignore_changes = [administrator_login_password_wo_version]
  }
}

resource "azurerm_mssql_database" "main" {
  name                 = "sqldb-poligpt-dev"
  server_id            = azurerm_mssql_server.main.id
  storage_account_type = "Local"

  lifecycle {
    prevent_destroy = true
  }
}

# Permite acesso de serviços da Azure (Databricks e indexador do AI Search).
resource "azurerm_mssql_firewall_rule" "allow_azure_services" {
  name             = "AllowAllWindowsAzureIps"
  server_id        = azurerm_mssql_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}
