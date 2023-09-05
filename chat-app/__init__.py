import os, secrets, uuid, markdown

from PIL import Image
from flask import Flask, render_template, request, session, url_for, redirect, send_from_directory, flash, escape
from llama_cpp import Llama

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        # app configuration
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'database.sqlite'),
        MODELS=os.path.join(app.instance_path, 'models'),
        IMAGE_FORMATS={'png', 'jpg', 'jpeg', 'webp'},
        UPLOAD_FOLDER=os.path.join(app.static_folder, 'uploads'),

        ALPACA_SYSTEM_STRING = """
        ### Instruction:
        Role play as the character described below. You always stay in character.
        You are {char_name} — {char_description}
        Remember, you always stay in character. You are the character described above.

        The circumstances of the roleplay are: {scenario}
        
        Repond to the dialogue below as your character would.

        ### Input:
        {history}
        {user_name}: {user_message}

        ### Response:
        {char_name}:""",

        DEFAULT_MODEL_NAME = "nous-hermes-llama2-13b.ggmlv3.q4_1.bin",
        DEFAULT_MAX_CONTEXT = 2048,
        DEFAULT_BATCH_SIZE = 256,
        DEFAULT_MAX_TOKENS = 256,
        DEFAULT_TEMPERATURE = 0.65,
        DEFAULT_TOP_P = 0.95,
        DEFAULT_TOP_K = 40,
        DEFAULT_REP_PEN = 1.1,
        DEFAULT_SEED = 1337,
        DEFAULT_GPU_OFFLOAD = 0,

        DEFAULT_CHAR_NAME = "Robbie",
        DEFAULT_EXAMPLES = "",
        DEFAULT_CHAR_DESCRIPTION = "A deeply philosophic and kind artificial intelligence with vast amounts of knowledge and wisdom.",
        DEFAULT_GREETING = "Hello, my name is Robbie. What would you like to discuss?",
        DEFAULT_SCENARIO = "A friendly A.I. is having a conversation with a curious human.",

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
            ),
            menu=request.args.get('show_menu')
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
            char_description = session.get('char_description', app.config['DEFAULT_CHAR_DESCRIPTION']), 
            examples = session.get('chat_examples', app.config['DEFAULT_EXAMPLES']), 
            user_name = session.get('user_name', app.config['DEFAULT_USER_NAME']),
            scenario = session.get('scenario', app.config['DEFAULT_SCENARIO']), 
            history = formatted_history,
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
                char_description = session.get('char_description', app.config['DEFAULT_CHAR_DESCRIPTION']), 
                examples = session.get('chat_examples', app.config['DEFAULT_EXAMPLES']), 
                user_name = session.get('user_name', app.config['DEFAULT_USER_NAME']),
                scenario = session.get('scenario', app.config['DEFAULT_SCENARIO']), 
                history = trimmed_history,
                user_message = user_message)
        
        bot_output = llm(full_context, stop=[f"{session.get('user_name', app.config['DEFAULT_USER_NAME'])}: ", "### Response:", "### Input:", "The following message is part of the current interaction; respond to it."], max_tokens=session.get('max_tokens', app.config['DEFAULT_MAX_TOKENS']), echo=False)
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
    
    @app.route('/update-user', methods=['POST',])
    def update_user():
        if request.form.get('user-name') != None:
            session['user_name'] = request.form['user-name']
        
        return redirect(url_for('index', show_menu='user'))

    @app.route('/update-char', methods=['POST',])
    def update_character():
        if request.form.get('reset-default') == '1':
            session['char_name'] = app.config['DEFAULT_CHAR_NAME']
            session['greeting'] = app.config['DEFAULT_GREETING']
            session['char_description'] = app.config['DEFAULT_CHAR_DESCRIPTION']
            session['scenario'] = app.config['DEFAULT_SCENARIO']
            session['chat_examples'] = app.config['DEFAULT_EXAMPLES']
            session['char_image'] = 'default.jpg'

            return redirect(url_for('index', show_menu='character'))

        if request.form.get('char-name') != None:
            session['char_name'] = request.form['char-name']
        if request.form.get('char-greeting') != None:
            session['greeting'] = request.form['char-greeting']
        if request.form.get('char-description'):
            session['char_description'] = request.form['char-description']
        if request.form.get('char-scenario') != None:
            session['scenario'] = request.form['char-scenario']
        if request.form.get('char-examples') != None:
            session['chat_examples'] = request.form['char-examples']
        
        character_icon = request.files.get('char-image')
        if character_icon != None and character_icon.filename != '':
            extension = character_icon.filename.rsplit('.', 1)[1].lower()
            if extension in app.config['IMAGE_FORMATS']:
                img = Image.open(character_icon)
                scale = 500 / min(img.size[0], img.size[1])
                img = img.resize((int(scale * img.size[0]), int(scale * img.size[1])))
                img = img.crop(box=(int(img.size[0] / 2) - 250, 0, 500 + int(img.size[0] / 2) - 250, 500))
                file_id = str(secrets.token_hex(8))
                session['char_image'] = '.'.join([file_id, extension])
                img.save(os.path.join(app.config['UPLOAD_FOLDER'], '.'.join([file_id, extension])))
            else:
                flash('File type not supported')
        
        return redirect(url_for('index', show_menu='character'))
            
    @app.route('/update-model', methods=['POST',])
    def update_model():
        pass

    return app