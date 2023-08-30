import os

from flask import Flask, render_template, request, session, url_for, redirect, send_from_directory
from llama_cpp import Llama

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        # app configuration
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'database.sqlite'),
        MODELS=os.path.join(app.instance_path, 'models'),

        ALPACA_SYSTEM_STRING = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ### Instruction:
        Role-play as the character that is described in the following lines. You always stay in character.
        Your name is {char_name}
        Your personality is as follows: {char_personality}
        You always stay in character. You are the character described above.

        The following is a set of example conversations to further inform you on how you should behave: 
        {chat_examples}

        Circumstances of the role-play for context: 
        {scenario}

        History of the role-play for context:
        {chat_history}

        Respond to the following message as your character would:

        ### Input:
        {user_message}

        ### Response:
        {char_name}:""",

        DEFAULT_MODEL_NAME = "model.bin",
        DEFAULT_MAX_CONTEXT = 2048,
        DEFAULT_BATCH_SIZE = 256,
        DEFAULT_MAX_TOKENS = 256,
        DEFAULT_TEMPERATURE = 0.65,
        DEFAULT_TOP_P = 0.95,
        DEFAULT_TOP_K = 40,
        DEFAULT_REP_PEN = 1.1,
        DEFAULT_SEED = 1337,
        DEFAULT_GPU_OFFLOAD = 0,

        DEFAULT_CHAR_NAME = "AI",
        DEFAULT_CHAR_PERSONALITY = "A helpful and kind AI assistant.",
        DEFAULT_GREETING = "Hello! What can I do for you?",
        DEFAULT_EXAMPLES = """AI: How may I help you User?
        User: I would love to hear a story.
        AI: Sure thing! What would you like the story to be about?
        User: Pirates
        AI: There was once an invisible ship aboard which some wicked pirates lived. 
            These pirates spent their days sailing the seas and oceans hunting for very valuable treasure – some hidden treasure that no one had ever been able to find.
            The pirates and their ship were invisible, and you could only see it if you were a pirate too. 
            It also meant that the pirates could get to all the hidden treasure before anyone else, for they wouldn’t leave a trace.
        """,
        DEFAULT_SCENARIO = "A friendly, enthusiastic AI is having a conversation with a curious human.",

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

    # root static overrides
    @app.route('/favicon.ico')
    def static_from_root():
        return send_from_directory(app.static_folder, request.path[1:])

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
                    session.get('greeting', app.config['DEFAULT_GREETING']),
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
            session["chat_history"] = [[None, session.get('greeting', app.config['DEFAULT_GREETING'])],]

        # generate bot message
        formatted_history = ''
        for m in session["chat_history"]:
            if m[0]:
                formatted_history += f"{session.get('user_name', app.config['DEFAULT_USER_NAME'])}: {m[0]}\n"
            if m[1]:
                formatted_history += f"{session.get('char_name', app.config['DEFAULT_CHAR_NAME'])}: {m[1]}\n"

        full_context = app.config['ALPACA_SYSTEM_STRING'].format(
            char_name = session.get('char_name', app.config['DEFAULT_CHAR_NAME']), 
            char_personality = session.get('char_personality', app.config['DEFAULT_CHAR_PERSONALITY']), 
            chat_examples = session.get('chat_examples', app.config['DEFAULT_EXAMPLES']), 
            user_name = session.get('user_name', app.config['DEFAULT_USER_NAME']),
            scenario = session.get('scenario', app.config['DEFAULT_SCENARIO']), 
            chat_history = formatted_history,
            user_message = user_message)
        
        while len(llm.tokenize(full_context.encode())) > session.get('max_context', app.config['DEFAULT_MAX_CONTEXT']):            
            session["chat_history"] = session["chat_history"][1:]
            trimmed_history = ''
            for m in session["chat_history"]:
                if m[0]:
                    trimmed_history += f"{session.get('user_name', app.config['DEFAULT_USER_NAME'])}: {m[0]}\n"
                if m[1]:
                    trimmed_history += f"{session.get('char_name', app.config['DEFAULT_CHAR_NAME'])}: {m[1]}\n"
            
            full_context = app.config['ALPACA_SYSTEM_STRING'].format(
                char_name = session.get('char_name', app.config['DEFAULT_CHAR_NAME']), 
                char_personality = session.get('char_personality', app.config['DEFAULT_CHAR_PERSONALITY']), 
                chat_examples = session.get('chat_examples', app.config['DEFAULT_EXAMPLES']), 
                user_name = session.get('user_name', app.config['DEFAULT_USER_NAME']),
                scenario = session.get('scenario', app.config['DEFAULT_SCENARIO']), 
                chat_history = trimmed_history,
                user_message = user_message)

        bot_output = llm(full_context, max_tokens=session.get('max_tokens', app.config['DEFAULT_MAX_TOKENS']), echo=False)
        bot_message = bot_output['choices'][0]['text']
        
        # add messages to history
        session["chat_history"] = session["chat_history"] + [[user_message, bot_message]]

        return {
            "message": bot_message
        }

    @app.route('/reset')
    def reset():
        if "chat_history" in session:
            session.pop("chat_history")
        return redirect(url_for('index'))

    return app