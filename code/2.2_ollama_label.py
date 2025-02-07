# curl -fsSL https://ollama.com/install.sh | sh
# pip install datasets ollama huggingface_hub

import os
import json
from datasets import load_dataset
import ollama
from ollama import chat
from ollama import ChatResponse
from huggingface_hub import login
from dotenv import load_dotenv
from model_common import SYSTEM_PROMPT, DEFINITIONS, ThoughtfulClassification


global MODEL
MODEL = "phi4"


def ollama_label(example):
    response: ChatResponse = chat(
        model=MODEL,
        format=ThoughtfulClassification.model_json_schema(),
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT,
            },
            {
                'role': 'user',
                'content': example['text'],
            },
        ],
        options={'temperature': 0.3}
    )
    parsed_response_content = json.loads(response.message.content)
    for response_key in parsed_response_content:
        response_value = parsed_response_content[response_key]
        if type(response_value) is list:
            for definition_key in DEFINITIONS.keys():
                example[definition_key] = definition_key in response_value
        else:
            example[response_key] = response_value

    return example


def main():
    # Login
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)

    # Pull model
    ollama.pull(MODEL)

    # Load data
    dataset = load_dataset('alex-miller/crs-2014-2023-housing-selection', split='train')

    # Label
    dataset = dataset.map(ollama_label)
    dataset.push_to_hub('alex-miller/crs-2014-2023-housing-labeled-phi4')


if __name__ == '__main__':
    main()
