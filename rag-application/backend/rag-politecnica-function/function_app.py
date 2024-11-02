import azure.functions as func
from chat_history import bp_history
from chat_completion import bp_chat

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Register chat completion function
app.register_blueprint(bp_chat)

# Register get history function
app.register_blueprint(bp_history)
