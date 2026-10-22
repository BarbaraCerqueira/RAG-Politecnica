from delta.tables import DeltaTable
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, max
from databricks.sdk.runtime import dbutils
from datetime import datetime
import time

spark = SparkSession.builder.getOrCreate()


# Função para listar todos os arquivos em um diretório e seus subdiretórios de primeiro nível
# Permite filtrar por formato de arquivo e timestamp de última modificação para obter apenas arquivos atualizados
def list_all_files(base_dir, file_format=None, min_modified_timestamp=None):
    files = []
    directories = dbutils.fs.ls(base_dir)
    
    for directory in directories:
        # Procurar os arquivos elegíveis dentro de cada subdiretório
        sub_files = dbutils.fs.ls(directory.path)
        for file_info in sub_files:
            if file_format is None or file_info.path.endswith(f".{file_format}"):
                if min_modified_timestamp is None or file_info.modificationTime > min_modified_timestamp:
                    files.append(file_info.path.replace("dbfs:", ""))

    print(f"Encontrados {len(files)} arquivos que satisfazem as condições de busca.")
    return files


# Função para salvar o ponto de progresso (timestamp) desde a última execução
def save_checkpoint(checkpoint_path, source_path, target_path):
    checkpoint_entry = [{
        "source_path": source_path,
        "target_path": target_path,
        "last_update": int(datetime.timestamp(datetime.now()) * 1000)
    }]
    checkpoint_df = spark.createDataFrame(checkpoint_entry)
    checkpoint_df.write.mode("append").format("parquet").save(checkpoint_path)
    print(f"Checkpoint salvo com sucesso em {checkpoint_path}")


# Função para obter o timestamp da última extração de uma determinada fonte
def get_checkpoint_by_source(checkpoint_path, source_path):
    try:
        last_extraction = int(
            spark.read.format("parquet").load(checkpoint_path)
            .filter(col("source_path") == source_path)
            .select(max("last_update"))
            .collect()[0][0]
        )
        print(f"Última extração encontrada: {last_extraction}")
    except:
        last_extraction = 0
        print("Não foi possível encontrar a tabela de checkpoints. Será feita uma extração completa.")
    finally:
        return last_extraction
    
    
# Função para obter o timestamp da última atualização de uma determinada tabela destino
def get_checkpoint_by_target(checkpoint_path, target_path):
    try:
        last_update = int(
            spark.read.format("parquet").load(checkpoint_path)
            .filter(col("target_path") == target_path)
            .select(max("last_update"))
            .collect()[0][0]
        )
        print(f"Última atualização encontrada: {last_update}")
    except:
        last_update = 0
        print("Não foi possível encontrar a tabela de checkpoints. Será feita uma carga completa.")
    finally:
        return last_update


# Função para realizar o Upsert (inserir ou atualizar) no Delta Lake
def upsert_to_delta_lake(df, delta_table_path, key_column):
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


# Função para realizar o merge de um dataframe Spark em uma tabela do banco SQL Server
def upsert_to_sql(
    df: DataFrame, schema_name: str, table_name: str, key: str,
    db_username: str, db_password: str, db_database: str, db_hostname: str
):
    # Nome completo das tabelas de interesse
    temp_table = f"{schema_name}.{table_name}_TEMP"
    target_table = f"{schema_name}.{table_name}"

    # Propriedades de conexão com o banco de dados via jdbc
    jdbc_url = f"jdbc:sqlserver://{db_hostname}:1433;database={db_database}"
    connection_properties = {
        "user" : db_username,
        "password" : db_password,
        "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver",
        "truncate": "true"
    }

    # Escrever os dados numa tabela temporária
    df.write.jdbc(
        url=jdbc_url, 
        table=temp_table,
        mode="overwrite",
        properties=connection_properties
    )
    
    # Estabelecer conexão com o banco de dados
    driver_manager = spark._sc._gateway.jvm.java.sql.DriverManager
    db_connection = driver_manager.getConnection(jdbc_url, db_username, db_password)
    
    try:
        merge_stmt = f"""
        MERGE INTO {target_table} AS target
        USING {temp_table} AS source
        ON target.{key} = source.{key}
        WHEN MATCHED THEN 
            UPDATE SET {", ".join(f"target.{col} = source.{col}" for col in df.columns)}
        WHEN NOT MATCHED THEN 
            INSERT ({ ", ".join(df.columns)})
            VALUES ({ ", ".join(f"source.{col}" for col in df.columns)});
        """
        
        # Executar o MERGE entre a tabela temporária e a tabela de destino
        statement = db_connection.createStatement()
        statement.execute(merge_stmt)
        
        # Excluir a tabela temporária
        drop_temp_table_sql = f"DROP TABLE {temp_table};"
        statement.execute(drop_temp_table_sql)
        
    finally:
        db_connection.close()


# Função para realizar o upsert no banco SQL Server com lógica de retry
def upsert_to_sql_with_retry(max_retries, delay, **upsert_kwargs):
    retries = 0
    while retries < max_retries:
        try:
            upsert_to_sql(**upsert_kwargs)
            print("Carga no Banco SQL realizada com sucesso.")
            return
        
        except Exception as e:
            retries += 1
            print(f"Tentativa {retries} falhou: {e}")
            if retries < max_retries:
                print(f"Aguardando {delay} segundos antes de tentar novamente...")
                time.sleep(delay)
            else:
                print("Número máximo de tentativas excedido. Carga no Banco SQL não realizada.")
                raise
