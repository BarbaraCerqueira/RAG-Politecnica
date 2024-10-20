import logging
import json
import azure.functions as func
from get_embeddings import get_embeddings

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.function_name(name="CustomOpenAIEmbeddings")
@app.route(route="custom_web_api/openai_embeddings")
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Request para geração de embeddings recebida.")

    try:
        # Parse do JSON recebido do Cognitive Search
        req_body = req.get_json()
        values = req_body.get("values")

        # Preparar a resposta com os embeddings
        response_values = []

        for item in values:
            record_id = item.get("recordId")
            text = item.get("data").get("ds_content_document")

            if text:
                embedding = get_embeddings(text)

                if embedding:
                    response_values.append({
                        "recordId": record_id,
                        "data": {
                            "vc_content_embedding": embedding
                        },
                        "errors": None,
                        "warnings": None
                    })
                else:
                    response_values.append({
                        "recordId": record_id,
                        "data": {},
                        "errors": [{"message": "Erro ao gerar embedding"}],
                        "warnings": None
                    })
            else:
                response_values.append({
                    "recordId": record_id,
                    "data": {},
                    "errors": [{"message": "O campo 'ds_content_document' está vazio"}],
                    "warnings": None
                })

        # Montar a resposta no formato esperado pelo Cognitive Search
        return func.HttpResponse(
            json.dumps({"values": response_values}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Erro ao processar requisição: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )