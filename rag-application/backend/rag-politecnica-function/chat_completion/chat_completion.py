import json
import logging
import azure.functions as func
from utils import get_chain

bp_chat = func.Blueprint()

@bp_chat.function_name(name="ChatCompletion")
@bp_chat.route(route="chatcompletion")
def chat_completion(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Obtém a mensagem do usuário
        req_body = req.get_json()
        user_message = req_body.get('message')
        conversation_history = req_body.get('history', [])
        text_generation_model = req_body.get('llm_model', 'text-embedding-3-large')

        # Inicializa o chain do LangChain
        chain = get_chain(conversation_history, text_generation_model)

        # Executa o chain com a mensagem do usuário
        response = chain.run(input=user_message)

        # Atualiza o histórico de conversação
        conversation_history.append({"user": user_message, "bot": response})

        # Retorna a resposta e o histórico atualizado
        return func.HttpResponse(
            json.dumps({"response": response, "history": conversation_history}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        return func.HttpResponse(
            f"Erro: {str(e)}",
            status_code=500
        )