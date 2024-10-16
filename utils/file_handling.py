from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from databricks.sdk.runtime import *

spark = SparkSession.builder.getOrCreate()

# Função para listar todos os arquivos de determinado formato em subdiretórios
def list_all_files(base_dir, file_format):
    files = []
    
    # Listar os subdiretórios (URLs)
    directories = dbutils.fs.ls(base_dir)
    
    # Percorrer os subdiretórios
    for directory in directories:
        # Procurar os arquivos elegíveis dentro de cada subdiretório
        sub_files = dbutils.fs.ls(directory.path)
        for file_info in sub_files:
            if file_info.path.endswith(f".{file_format}"):
                files.append(file_info.path.replace("dbfs:", ""))
    
    return files

# Função para realizar o Upsert (inserir ou atualizar) no Delta Lake
def upsert_to_delta_lake(df, delta_table_path, key_column="file_path"):
    # Verificar se a tabela Delta já existe
    if DeltaTable.isDeltaTable(spark, delta_table_path):
        delta_table = DeltaTable.forPath(spark, delta_table_path)
        
        # Realizar o merge com base no key_column
        delta_table.alias("old") \
            .merge(
                df.alias("new"),
                f"old.{key_column} = new.{key_column}"
            ) \
            .whenMatchedUpdateAll() \
            .whenNotMatchedInsertAll() \
            .execute()

        print("Tabela mesclada no caminho:", delta_table_path)
    else:
        # Se a tabela não existir, cria a tabela Delta
        df.write.format("delta").mode("overwrite").save(delta_table_path)
        print("Tabela criada no caminho:", delta_table_path)

