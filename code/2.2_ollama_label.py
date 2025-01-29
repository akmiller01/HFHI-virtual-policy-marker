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


global MODEL
MODEL = "mistral:7b-instruct-v0.3-q3_K_M"

global SYSTEM_PROMPT
global URBAN_SYSTEM_PROMPT
global DEFINITIONS
global FORMAT
SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies text. "
    "You are classifying whether the text {}. "
    "You respond in JSON format, first giving thoughts in as few words as needed about whether the text matches the definition above in the 'thoughts' key and "
    "then giving your answer in the 'answer' key."
)
URBAN_SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies text. "
    "You are classifying whether the text explicitly describes activities in specific urban locations, specific rural locations, both urban and rural locations, or neither."
    "You respond in JSON format, first giving thoughts in as few words as needed about whether the text matches the definitions above in the 'thoughts' key and "
    "then giving your answer in the 'answer' key. Possible answer choices are 'Urban', 'Rural', 'Both', or 'Neither'."
)
DEFINITIONS = {
    "housing": "directly or indirectly relates to any of the following: housing, housing policy, housing finance, habitability, tents for the homeless, encampments for the homeless, homeless shelters, emergency shelters, refugee shelters, refugee camps, temporary supportive housing, housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, neighborhood integration, community land trusts, cooperative housing, public housing, subsidized home-rental, subsidized mortgages, rent-to-own housing, or market-rate housing",
    "homelessness": "explicitly describes tents for the homeless, encampments for the homeless, or homeless shelters",
    "transitional": "explicitly describes emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
    "incremental": "explicitly describes housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration",
    "social": "explicitly describes community land trusts, cooperative housing, or public housing",
    "market": "explicitly describes subsidized home-rental, subsidized mortgages, rent-to-own housing, or market-rate housing",
}
FORMAT = {
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "string"
        },
        "answer": {
            "type": "boolean"
        }
    },
    "required": [
        "thoughts",
        "answer"
    ]
}
URBAN_FORMAT = {
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "string"
        },
        "answer": {
            "type": "string"
        }
    },
    "required": [
        "thoughts",
        "answer"
    ]
}


def ollama_label(example):
    for key in DEFINITIONS.keys():
        definition = DEFINITIONS[key]
        definition_system_prompt = SYSTEM_PROMPT.format(definition)
        response: ChatResponse = chat(
            model=MODEL,
            format=FORMAT,
            messages=[
                {
                    'role': 'system',
                    'content': definition_system_prompt,
                },
                {
                    'role': 'user',
                    'content': example['text'],
                },
            ]
        )
        parsed_response_content = json.loads(response.message.content)
        for response_key in parsed_response_content:
            definition_response_key = f"{key}_{response_key}"
            example[definition_response_key] = parsed_response_content[response_key]
        if example['housing_answer'] == False:
            for other_key in list(DEFINITIONS.keys())[1:]:
                example[f"{other_key}_thoughts"] = ""
                example[f"{other_key}_answer"] = False
            example["urban_rural_thoughts"] = ""
            example["urban_rural_answer"] = "Neither"
            break
    if example['housing_answer'] == True:
        key = "urban_rural"
        response: ChatResponse = chat(
            model=MODEL,
            format=URBAN_FORMAT,
            messages=[
                {
                    'role': 'system',
                    'content': URBAN_SYSTEM_PROMPT,
                },
                {
                    'role': 'user',
                    'content': example['text'],
                },
            ]
        )
        parsed_response_content = json.loads(response.message.content)
        for response_key in parsed_response_content:
            definition_response_key = f"{key}_{response_key}"
            example[definition_response_key] = parsed_response_content[response_key]
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
    dataset.push_to_hub('alex-miller/crs-2014-2023-housing-labeled')


if __name__ == '__main__':
    main()
