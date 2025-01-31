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
from sector_dict import SECTORS


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
    "For additional context, the donor has chosen the '{}' sector for this activity, but that does not preclude it from also belonging to the housing sector.\n"
    "Think carefully and do not jump to conclusions: ground your response in the given text.\n"
    "Respond in JSON format, first giving your complete thoughts about all the possible matches with the above classes and definitions in the 'thoughts' key "
    "and then listing all of the classes that match in the 'classifications' key."
)
DEFINITIONS = {
    "Housing": "describes the housing sector, including but not limited to: provision of housing, provision of shelter in emergencies, upgrading inadequate housing, construction of housing, urban development, housing policy, technical assistance for housing, or finance for housing",
    "Homelessness": "describes housing support for the homeless, including but not limited to: tents for the homeless, encampments for the homeless, or homeless shelters",
    "Transitional": "describes transitional housing, including but not limited to: emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
    "Incremental": "describes incremental housing, including but not limited to: housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration",
    "Social": "describes social housing, including but not limited to: community land trusts, cooperative housing, or public housing",
    "Market": "describes market-based housing support, including but not limited to: home-rental, mortgages, rent-to-own housing, or market-rate housing",
    "Urban": "describes specific urban locations",
    "Rural": "describes specific rural locations"
}
SYSTEM_PROMPT = SYSTEM_PROMPT.format(
    "\n".join([f'- {key}' for key in DEFINITIONS.keys()]),
    "\n".join([f'- {key}: when the text {value}' for key, value in DEFINITIONS.items()]),
    "{}"
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
                'content': SYSTEM_PROMPT.format(SECTORS[str(example['sector_code'])]),
            },
            {
                'role': 'user',
                'content': example['text'],
            },
        ],
        options={'temperature': 0.2}
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
    unique_sectors = [str(sector) for sector in list(set(dataset['sector_code']))]
    missing_sectors = [sector for sector in unique_sectors if not sector in SECTORS]
    if len(missing_sectors) > 0:
        raise Exception(f"Please add the following sector codes to code/sector_dict.py:\n{"\n".join(missing_sectors)}")

    # Label
    dataset = dataset.map(ollama_label)
    dataset.push_to_hub('alex-miller/crs-2014-2023-housing-labeled-phi4')


if __name__ == '__main__':
    main()
