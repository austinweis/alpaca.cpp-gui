#!/bin/sh
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export MODEL_PATH=$1 # take first argument as relative path to gguf model

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python -m venv $SCRIPT_DIR/.venv
fi
source $SCRIPT_DIR/.venv/bin/activate

echo "Updating packages..."
python -m pip install -r $SCRIPT_DIR/requirements.txt

echo "Starting server..."
flask --app $SCRIPT_DIR/chat-app run -h 0.0.0.0