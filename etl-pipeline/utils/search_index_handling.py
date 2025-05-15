from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import SearchIndexer
from datetime import datetime
import time


# Obter o cliente do indexador
def get_indexer_client(indexer_name, azure_search_endpoint, azure_search_admin_key):
    azure_search_credential = AzureKeyCredential(azure_search_admin_key)
    indexer_client = SearchIndexerClient(
        azure_search_endpoint, 
        azure_search_credential
    )
    return indexer_client


# Roda o indexador do Azure Search AI com até max_retries tentativas em casos de falha
def run_indexer_with_retry(indexer_client, indexer_name, max_retries=3, delay_check=30, timeout_total=300):

    for attempt in range(1, max_retries + 1):
        print(f"\nTentativa {attempt}: rodando indexador {indexer_name}...")
        
        # Obter o último end_time antes de iniciar nova execução - ele será usado para detectar se a execução terminou
        previous_status = indexer_client.get_indexer_status(indexer_name).as_dict()
        previous_end_time = None
        if previous_status.get("last_result"):
            previous_end_time = previous_status["last_result"].get("end_time")
        
        indexer_client.run_indexer(indexer_name)
        print("Aguardando nova execução finalizar...")

        start_wait = time.time()
        while time.time() - start_wait < timeout_total:
            time.sleep(delay_check)
            current_status = indexer_client.get_indexer_status(indexer_name).as_dict()
            last_result = current_status.get("last_result", [])
            
            if not last_result:
                continue

            latest_end_time = last_result.get("end_time")

            # Seguir apenas no caso de o end_time ter mudado — significa que a execução terminou
            if latest_end_time and latest_end_time != previous_end_time:
                status = last_result.get("status")
                print(f"Execução finalizada com status: {status}")
                
                if status == "success":
                    print("Indexador rodou com sucesso!")
                    return True
                else:
                    print(f"Indexador falhou com: {last_result.get('error_message', 'Erro desconhecido')}")
                    break

        print(f"Timeout ou falha detectada. Nova tentativa será feita caso haja tentativas restantes...")
        time.sleep(10)

    print("Todas as tentativas falharam.")
    return False
