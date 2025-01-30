# curl -fsSL https://ollama.com/install.sh | sh
# pip install datasets ollama huggingface_hub

import os
import json
from datasets import load_dataset
import ollama
from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel
from enum import Enum
from huggingface_hub import login
from dotenv import load_dotenv


global MODEL
MODEL = "mistral:7b-instruct-v0.3-q3_K_M"

global SYSTEM_PROMPT
global URBAN_SYSTEM_PROMPT
global DEFINITIONS
SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies text. "
    "You are classifying whether the text {}. "
    "Do not jump to conclusions: ground your response on the given text. "
    "You respond in JSON format, first giving your thoughts about whether the text matches the definition above in the 'thoughts' key and "
    "then giving your answer in the 'answer' key."
)
URBAN_SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies text. "
    "You are classifying whether the text explicitly describes activities in specific urban locations, specific rural locations, both urban and rural locations, or neither."
    "Do not jump to conclusions: ground your response on the given text. "
    "You respond in JSON format, first giving your thoughts about whether the text matches the definitions above in the 'thoughts' key and "
    "then giving your answer in the 'answer' key. Possible answer choices are 'Urban', 'Rural', 'Both', or 'Neither'."
)
DEFINITIONS = {
    "housing": "describes the housing sector, including but not limited to: provision of housing, provision of shelter in emergencies, improving the quality of life in inadequate housing, construction of housing, urban development, housing policy, technical assistance for housing, or finance for housing",
    "homelessness": "explicitly describes tents for the homeless, encampments for the homeless, or homeless shelters",
    "transitional": "explicitly describes emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
    "incremental": "explicitly describes housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration",
    "social": "explicitly describes community land trusts, cooperative housing, or public housing",
    "market": "explicitly describes home-rental, mortgages, rent-to-own housing, or market-rate housing",
}
class ThoughtfulClassification(BaseModel):
    thoughts: str
    answer: bool
class LocationType(Enum):
    U = 'Urban'
    R = 'Rural'
    B = 'Both'
    N = 'Neither'
class ThoughtfulLocationClassification(BaseModel):
    thoughts: str
    answer: LocationType


def ollama_label(example):
    for key in DEFINITIONS.keys():
        definition = DEFINITIONS[key]
        definition_system_prompt = SYSTEM_PROMPT.format(definition)
        response: ChatResponse = chat(
            model=MODEL,
            format=ThoughtfulClassification.model_json_schema(),
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
    key = "urban_rural"
    response: ChatResponse = chat(
        model=MODEL,
        format=ThoughtfulLocationClassification.model_json_schema(),
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
