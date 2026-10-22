# Montar o objeto de acesso ao Azure Data Lake Storage no Azure Databricks usando Service Principal

configs = {"fs.azure.account.auth.type": "OAuth",
          "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
          "fs.azure.account.oauth2.client.id": "24229ad4-1abf-49a9-afca-980cd0c17874",
          "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope="poligpt-secret-scope",key="service-principal-secret"),
          "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/e27c2200-7502-4541-b4ab-c552838a6b21/oauth2/token"}

dbutils.fs.mount(
  source = "abfss://control@dlspoligptdev.dfs.core.windows.net/",
  mount_point = "/mnt/adlscontrol",
  extra_configs = configs)

dbutils.fs.mount(
  source = "abfss://raw@dlspoligptdev.dfs.core.windows.net/",
  mount_point = "/mnt/adlsraw",
  extra_configs = configs)

dbutils.fs.mount(
  source = "abfss://trusted@dlspoligptdev.dfs.core.windows.net/",
  mount_point = "/mnt/adlstrusted",
  extra_configs = configs)

dbutils.fs.mount(
  source = "abfss://refined@dlspoligptdev.dfs.core.windows.net/",
  mount_point = "/mnt/adlsrefined",
  extra_configs = configs)
