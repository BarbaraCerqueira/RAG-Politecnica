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

        # Pegar os parâmetros do modelo e dimensões dos headers
        model = req.headers.get("model", "text-embedding-3-large")
        dimensions = int(req.headers.get("dimensions", 1024))

        # Preparar uma lista de textos e seus recordIds
        texts = []
        record_ids = []
        for item in values:
            record_id = item.get("recordId")
            text = item.get("data").get("text")
            if text:
                texts.append(text)
                record_ids.append(record_id)

        # Gerar os embeddings para todos os textos em batch
        embeddings = get_embeddings(texts, model, dimensions)

        # Preparar a resposta com os embeddings mapeados de volta aos recordIds
        response_values = []
        for record_id, embedding in zip(record_ids, embeddings):
            response_values.append({
                "recordId": record_id,
                "data": {
                    "embedding": embedding
                },
                "errors": None,
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