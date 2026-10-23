# ---------------------------------------------------------------------------
# Sobre a senha do administrador do SQL:
#
# A senha mora só no Key Vault (secret "sql-admin-password"). Quem a usa:
#   - Databricks: o notebook da camada refined grava no banco (via secret scope)
#   - AI Search: a fonte de dados lê do banco (connection string do secret "sqldb-connection-string")
#
# O Terraform precisa informar a senha ao declarar o servidor, mas não deve guardá-la:
#   - "ephemeral": o Terraform lê o secret do Key Vault durante a execução e descarta
#     o valor ao terminar; ele não é salvo no plano nem no state.
#   - "write-only" (atributo *_wo): o valor é enviado para a Azure, mas o Terraform não
#     o grava no state nem o compara depois. Por isso ele só é reenviado quando o número
#     de versão (*_wo_version) muda.
#
# Para trocar a senha: gravar a nova no Key Vault (e atualizar "sqldb-connection-string"),
# remover o ignore_changes abaixo e incrementar administrator_login_password_wo_version.
# ---------------------------------------------------------------------------
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

  administrator_login_password_wo         = ephemeral.azurerm_key_vault_secret.sql_admin_password.value
  administrator_login_password_wo_version = 1

  lifecycle {
    prevent_destroy = true
    # Na importação o servidor não tinha número de versão da senha; sem isto, o Terraform
    # reenviaria a senha (a mesma do Key Vault) no próximo apply. Hoje, na prática,
    # o Terraform não altera a senha do servidor sql.
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
