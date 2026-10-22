terraform {
  required_version = ">= 1.6"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.80"
    }
  }

  # State remoto no storage account dedicado (rg-poligpt-tfstate).
  # Autenticação via Entra ID (az login), sem chave de acesso.
  backend "azurerm" {
    resource_group_name  = "rg-poligpt-tfstate"
    storage_account_name = "stpoligpttfstate"
    container_name       = "tfstate"
    key                  = "poligpt-dev.tfstate"
    use_azuread_auth     = true
  }
}

provider "azurerm" {
  features {
    key_vault {
      # Nunca apagar definitivamente um Key Vault ou secret ao destruir.
      purge_soft_delete_on_destroy    = false
      recover_soft_deleted_key_vaults = true
    }
  }

  subscription_id = var.subscription_id

  # A assinatura de estudante já tem os providers necessários registrados.
  resource_provider_registrations = "none"
}
