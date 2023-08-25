# pygmalion format 
'''
[CHARACTER]'s Persona: [A few sentences about the character you want the model to play]
<START>
[DIALOGUE HISTORY]
You: [User's input message here]
[CHARACTER]:
'''

import os, sys
from llama_cpp import Llama

# character settings
char_name = ""
char_description = ""
first_mes = ""
examples = ""

# model settings
path = "./models/ggml-model-q5_1.bin"
max_context = 2048
batch_size = 256
llm = Llama(model_path=path, n_ctx=max_context, n_batch=batch_size)

# generation settings
max_tokens = 192
temperature = 0.65
rep_pen = 1.1

# variables
history = first_mes
token_count = len(llm.tokenize(bytes(f"{char_name}'s: Persona: {char_description}\n{examples}\n<START>\n{history}", 'utf-8')))

while 1:
    print(history)
    print(f"Token Count: {token_count}")
    user_input = input("You: ")

    # remove context if token count is too high
    token_count = len(llm.tokenize(bytes(f"{char_name}'s: Persona: {char_description}\n{examples}\n<START>\n{history}\nYou: {user_input}\n{char_name}: ", 'utf-8')))
    if (token_count > max_context):
        history = '\n'.join(history.split('\n')[1:])

    bot_output = llm(f"{char_name}'s: Persona: {char_description}\n{examples}\n<START>\n{history}\nYou: {user_input}\n{char_name}: " \
        ,max_tokens=max_tokens, temperature=temperature, repeat_penalty=rep_pen, stop=["You: "], echo=False)

    history = f"{history}\nYou: {user_input}\n{char_name}: {bot_output['choices'][0]['text']}"