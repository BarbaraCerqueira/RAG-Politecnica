variable "subscription_id" {
  description = "ID da assinatura Azure (Azure for Students)."
  type        = string
}

variable "tenant_id" {
  description = "ID do tenant (diretório) do Entra ID."
  type        = string
}

variable "location" {
  description = "Região dos recursos."
  type        = string
  default     = "eastus2"
}

variable "admin_object_id" {
  description = "Object ID da usuária administradora (acesso total aos secrets do Key Vault)."
  type        = string
}

variable "pipeline_sp_object_id" {
  description = "Object ID do service principal usado pelo Databricks para acessar o Data Lake (service-principal-poligpt)."
  type        = string
}

variable "databricks_app_object_id" {
  description = "Object ID do enterprise app global 'AzureDatabricks' no tenant (usado pelo secret scope baseado em Key Vault)."
  type        = string
}

variable "chat_cors_allowed_origins" {
  description = "Origens autorizadas a chamar a API do chat pelo navegador."
  type        = list(string)
}
