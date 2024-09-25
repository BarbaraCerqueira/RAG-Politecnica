from datetime import datetime
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

INGESTION_TABLE_PATH = "/mnt/adlscontrol/ingestion_logs/"
CATALOG = "control"
SCHEMA = "logs"
TABLE = "ingestion_logs"

# def check_table_exists(table_path):
#     """
#     Verifica se a tabela já existe no caminho especificado.

#     Args:
#         table_path (str): Caminho onde a tabela deveria estar.

#     Returns:
#         bool: Retorna True se a tabela existir, caso contrário False.
#     """
#     try:
#         # Tentativa de ler a tabela no Delta Lake
#         spark.read.format("delta").load(table_path)
#         return True
#     except:
#         return False

def log_ingestion_event(run_id, status, log_message, nr_processed_urls):
    """
    Registra um evento de ingestão na tabela de controle de ingestão.

    Args:
        run_id (str): Run ID da job de ingestão.
        status (str): Status da operação ('Sucesso', 'Falha', etc.).
        log_message (str): Mensagem de log.
        nr_processed_urls (int): Número de URLs processadas.
    
    Returns:
        None
    """
    try:
        # if not check_table_exists(INGESTION_TABLE_PATH):
        #     table_full_name = f"{CATALOG}.{SCHEMA}.{TABLE}"

        #     create_table_query = f"""
        #     CREATE TABLE IF NOT EXISTS {table_full_name} 
        #     USING DELTA 
        #     LOCATION '{f'{INGESTION_TABLE_PATH}'}'
        #     """

        #     # Criar a tabela no catálogo
        #     spark.sql(create_table_query)

        datetime_now = datetime.now()
        new_entry = [(run_id, datetime_now, status, log_message, nr_processed_urls)]
        df = spark.createDataFrame(new_entry, ["id_job_run", "dt_ingestion", "ds_status", "ds_log_message", "nr_processed_urls"])
        df.write.format("delta").option("header", "true").mode("append").save(INGESTION_TABLE_PATH)
        print(f"Novo log de ingestão salvo com ID: {run_id}")        
    
    except Exception as e:
        print(f"Erro ao registrar evento de ingestão: {e}")
