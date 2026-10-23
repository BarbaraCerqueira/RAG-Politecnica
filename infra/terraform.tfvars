# Valores não sensíveis do ambiente dev.
# Segredos (chaves de API, senhas) NUNCA entram aqui: ficam no Key Vault.

subscription_id = "928be6ea-abeb-46e9-9ffb-251008fa9e32"
tenant_id       = "e27c2200-7502-4541-b4ab-c552838a6b21"
location        = "eastus2"

admin_object_id          = "32ef9b3d-a892-421e-a72e-7eb3c2d55f0b"
pipeline_sp_object_id    = "155c4ba9-d6b0-4c3a-affe-11f3faa6e1b6"
databricks_app_object_id = "65bed51a-69dc-46ca-b10f-97e7adea4486"

chat_cors_allowed_origins = [
  "http://localhost:5173",
  "https://portal.azure.com",
]
