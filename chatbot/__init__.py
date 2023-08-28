import os

from flask import Flask, render_template, request, session
from llama_cpp import Llama

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        # app configuration
        SECRET_KEY='1',
        DATABASE=os.path.join(app.instance_path, 'database.sqlite'),
        MODELS=os.path.join(app.instance_path, 'models'),

        DEFAULT_MODEL_NAME = "model.bin",
        DEFAULT_MAX_CONTEXT = 1028,
        DEFAULT_BATCH_SIZE = 256,
        DEFAULT_MAX_TOKENS = 128,
        DEFAULT_TEMPERATURE = 0.65,
        DEFAULT_TOP_P = 0.95,
        DEFAULT_TOP_K = 40,
        DEFAULT_REP_PEN = 1.1,
        DEFAULT_SEED = -1,

        DEFAULT_BOT_NAME = "AI",
        DEFAULT_SYSTEM_PROMPT = "The following is a conversation between a helpful AI assistant and a user of it:",
        DEFAULT_FIRST_MESSAGE = "Hello! How may I help you?",

        DEFAULT_USER_NAME = "User",
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)
    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    llm = Llama(
        model_path=os.path.join(app.config["MODELS"], app.config['DEFAULT_MODEL_NAME']), 
        n_ctx=app.config['DEFAULT_MAX_CONTEXT'], 
        n_batch=app.config['DEFAULT_BATCH_SIZE'],
        n_threads=None,
        seed=app.config['DEFAULT_SEED'],
        verbose=False
    )

    @app.route('/')
    def index():
        print(session.get("chat_history"))
        return render_template(
            'chat.html',
            history=session.get(
                "chat_history",
                [
                    [
                    None,
                    session.get("first_message", app.config['DEFAULT_FIRST_MESSAGE']), 
                    ]
                    ,
                ]
            )
        )

    @app.route('/chat', methods=['POST',])
    def chat():
        user_message = request.form.get("message")

        # create "chat_history" if not initialized already
        if 'chat_history' not in session:
            session["chat_history"] = [[None, session.get('first_message', app.config['DEFAULT_FIRST_MESSAGE'])],]
        
        # generate bot message
        formatted_history = ''
        for m in session["chat_history"]:
            if m[0]:
                formatted_history += session.get('user_name', app.config['DEFAULT_USER_NAME'])
                formatted_history += f': {m[0]}\n'
            if m[1]:
                formatted_history += session.get('bot_name', app.config['DEFAULT_BOT_NAME'])
                formatted_history += f': {m[1]}\n'
        formatted_history += session.get('user_name', app.config['DEFAULT_USER_NAME'])
        formatted_history += f': {user_message}\n'

        full_context = f"{session.get('system_prompt', app.config['DEFAULT_SYSTEM_PROMPT'])}\n\n{formatted_history}{session.get('bot_name', app.config['DEFAULT_BOT_NAME'])}: "
        bot_output = llm(full_context, stop=[f"\n{session.get('user_name', app.config['DEFAULT_USER_NAME'])}: "], echo=False)
        bot_message = bot_output['choices'][0]['text']
        print(bot_output['choices'])
        
        # add messages to history
        session["chat_history"].append([user_message, bot_message])
        session["chat_history"] = session["chat_history"] # update history

        return {
            "message": bot_message
        }

    return app
