# curl -fsSL https://ollama.com/install.sh | sh
# pip install datasets ollama huggingface_hub

import os
import json
from datasets import load_dataset
import ollama
from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel
from typing import Literal
from huggingface_hub import login
from dotenv import load_dotenv


global MODEL
MODEL = "phi4"

global SYSTEM_PROMPT
global URBAN_SYSTEM_PROMPT
global DEFINITIONS
SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies development and humanitarian activity titles and descriptions.\n"
    "You are looking for matches with an expanded definition of the housing sector that encompasses a continuum defined by the possible classes below.\n"
    "The possible classes you are looking for are:\n"
    "{}\n"
    "The definitions of each possible class are:\n"
    "{}\n"
    "Think carefully and do not jump to conclusions: ground your response on the given text.\n"
    "Respond in JSON format, first giving your complete thoughts about all the possible matches with the above classes and definitions in the 'thoughts' key "
    "and then listing all of the classes that match in the 'classifications' key."
)
DEFINITIONS = {
    "Housing": "describes the housing sector, including but not limited to: provision of housing, provision of shelter in emergencies, improving the quality of life in inadequate housing, construction of housing, urban development, housing policy, technical assistance for housing, or finance for housing",
    "Homelessness": "explicitly describes tents for the homeless, encampments for the homeless, or homeless shelters",
    "Transitional": "explicitly describes emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
    "Incremental": "explicitly describes housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration",
    "Social": "explicitly describes community land trusts, cooperative housing, or public housing",
    "Market": "explicitly describes home-rental, mortgages, rent-to-own housing, or market-rate housing",
    "Urban": "explicitly describes activities in specific urban locations",
    "Rural": "explicitly describes activities in specific rural locations"
}
SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    "\n".join([f'- {key}' for key in DEFINITIONS.keys()]),
    "\n".join([f'- {key}: when the text {value}' for key, value in DEFINITIONS.items()]),
)

class ThoughtfulClassification(BaseModel):
    thoughts: str
    classifications: list[Literal['Housing', 'Homelessness', 'Transitional', 'Incremental', 'Social', 'Market', 'Urban', 'Rural']]


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
        ]
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
