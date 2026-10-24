resource "azurerm_databricks_workspace" "main" {
  name                = "dbw-poligpt-dev-eastus2"
  resource_group_name = azurerm_resource_group.main.name
  location            = var.location
  sku                 = "standard"

  lifecycle {
    prevent_destroy = true
    # TEMPORÁRIO: atributo sem valor no recurso importado (o padrão já é true).
    # Removido na etapa de migração para o plano premium.
    ignore_changes = [public_network_access_enabled]
  }
}

locals {
  # Bibliotecas do pipeline, lidas do requirements.txt (fonte única para o cluster interativo e o job).
  pipeline_libraries = [
    for line in split("\n", file("${path.module}/../pipeline/requirements.txt")) :
    trimspace(line) if trimspace(line) != "" && !startswith(trimspace(line), "#")
  ]

  # Configuração comum dos clusters: single node, DBR 15.4 LTS, sem Unity Catalog (workspace standard).
  cluster_spark_version = "15.4.x-scala2.12"
  cluster_node_type     = "Standard_F4"
  # O job usa uma VM com os mesmos 4 cores (limite da cota), mas com 32 GB de memória em vez de 8 trusted_poli_consolidated_html
  # o OCR do Tesseract em vários processos esgota a memória da F4.
  pipeline_node_type = "Standard_E4ds_v4"
  cluster_spark_conf = {
    "spark.databricks.cluster.profile" = "singleNode"
    "spark.master"                     = "local[*, 4]"
  }
  cluster_custom_tags = {
    "Projeto"       = "poligpt"
    "ResourceClass" = "SingleNode"
  }
  cluster_spark_env_vars = {
    # Onde o PyMuPDF encontra os modelos de idioma do Tesseract (instalados pelo init script).
    "TESSDATA_PREFIX" = "/usr/share/tesseract-ocr/4.00/tessdata"
    # Uma thread por OCR: o notebook já paraleliza com vários processos, e threads internas
    # do Tesseract disputariam os mesmos núcleos (recomendação da documentação do Tesseract).
    "OMP_THREAD_LIMIT" = "1"
  }

  pipeline_repo_url = "https://github.com/BarbaraCerqueira/poligpt.git"
}

# ---------------------------------------------------------------------------
# Secret scope ligado ao Key Vault: os notebooks leem os segredos com dbutils.secrets.get()
# ---------------------------------------------------------------------------
resource "databricks_secret_scope" "main" {
  name = "poligpt-secret-scope"

  keyvault_metadata {
    resource_id = azurerm_key_vault.main.id
    dns_name    = azurerm_key_vault.main.vault_uri
  }
}

# ---------------------------------------------------------------------------
# Init script que instala o Tesseract OCR nos clusters (fonte: pipeline/setup/init-scripts/)
# ---------------------------------------------------------------------------
resource "databricks_workspace_file" "install_tesseract" {
  source = "${path.module}/../pipeline/setup/init-scripts/install_tesseract.sh"
  path   = "/Shared/poligpt/init-scripts/install_tesseract.sh"
}

# ---------------------------------------------------------------------------
# Git folder com o repositório, usado para desenvolvimento interativo
# ---------------------------------------------------------------------------
resource "databricks_repo" "poligpt" {
  url    = local.pipeline_repo_url
  path   = "/Repos/barbaracerqueira9901@gmail.com/poligpt"
  branch = "main"
}

# ---------------------------------------------------------------------------
# Cluster interativo para desenvolvimento
# ---------------------------------------------------------------------------
resource "databricks_cluster" "interactive" {
  cluster_name            = "poligpt-cluster"
  kind                    = "CLASSIC_PREVIEW"
  is_single_node          = true
  spark_version           = local.cluster_spark_version
  node_type_id            = local.cluster_node_type
  driver_node_type_id     = local.cluster_node_type
  num_workers             = 0
  autotermination_minutes = 30
  data_security_mode      = "NONE"
  runtime_engine          = "STANDARD"
  enable_elastic_disk     = true
  spark_conf              = local.cluster_spark_conf
  spark_env_vars          = local.cluster_spark_env_vars
  custom_tags             = local.cluster_custom_tags

  azure_attributes {
    availability       = "ON_DEMAND_AZURE"
    first_on_demand    = 1
    spot_bid_max_price = -1
  }

  init_scripts {
    workspace {
      destination = databricks_workspace_file.install_tesseract.path
    }
  }

  dynamic "library" {
    for_each = local.pipeline_libraries
    content {
      pypi {
        package = library.value
      }
    }
  }
}

# ---------------------------------------------------------------------------
# Job do pipeline: lê os notebooks direto da main no GitHub e roda num cluster
# criado só durante a execução (job cluster), com as bibliotecas instaladas.
# ---------------------------------------------------------------------------
locals {
  # task_key => [caminho do notebook no repositório, dependências]
  pipeline_tasks = {
    sitemap_extraction             = ["pipeline/01_control/sitemap_extraction", []]
    raw_data_ingestion             = ["pipeline/02_raw/raw_data_ingestion", ["sitemap_extraction"]]
    trusted_poli_consolidated_html = ["pipeline/03_trusted/trusted_poli_consolidated_html", ["raw_data_ingestion"]]
    trusted_poli_consolidated_pdf  = ["pipeline/03_trusted/trusted_poli_consolidated_pdf", ["raw_data_ingestion"]]
    tb_poli_knowledge_base         = ["pipeline/04_refined/tb_poli_knowledge_base", ["trusted_poli_consolidated_html", "trusted_poli_consolidated_pdf"]]
    run_search_indexer             = ["pipeline/05_indexing/run_search_indexer", ["tb_poli_knowledge_base"]]
  }
}

resource "databricks_job" "pipeline" {
  name                = "poligpt_workflow"
  max_concurrent_runs = 1

  tags = {
    "Projeto" = "PoliGPT"
  }

  git_source {
    url      = local.pipeline_repo_url
    provider = "gitHub"
    branch   = "main"
  }

  # Toda semana, sábado às 03:00. Pausado até o pipeline ser validado de novo.
  schedule {
    quartz_cron_expression = "23 0 3 ? * Sat"
    timezone_id            = "America/Sao_Paulo"
    pause_status           = "PAUSED"
  }

  queue {
    enabled = true
  }

  job_cluster {
    job_cluster_key = "pipeline_cluster"

    new_cluster {
      spark_version       = local.cluster_spark_version
      node_type_id        = local.pipeline_node_type
      driver_node_type_id = local.pipeline_node_type
      num_workers         = 0
      data_security_mode  = "NONE"
      runtime_engine      = "STANDARD"
      enable_elastic_disk = true
      spark_conf          = local.cluster_spark_conf
      spark_env_vars      = local.cluster_spark_env_vars
      custom_tags         = local.cluster_custom_tags

      azure_attributes {
        availability       = "ON_DEMAND_AZURE"
        first_on_demand    = 1
        spot_bid_max_price = -1
      }

      init_scripts {
        workspace {
          destination = databricks_workspace_file.install_tesseract.path
        }
      }
    }
  }

  dynamic "task" {
    for_each = local.pipeline_tasks
    content {
      task_key        = task.key
      job_cluster_key = "pipeline_cluster"

      notebook_task {
        notebook_path = task.value[0]
        source        = "GIT"
      }

      dynamic "depends_on" {
        for_each = task.value[1]
        content {
          task_key = depends_on.value
        }
      }

      dynamic "library" {
        for_each = local.pipeline_libraries
        content {
          pypi {
            package = library.value
          }
        }
      }
    }
  }
}
