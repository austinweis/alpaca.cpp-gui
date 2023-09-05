# supply path to gguf model as first argument


export MODEL_PATH=$1
source .venv/bin/activate
flask --app chat-app run -h 0.0.0.0
