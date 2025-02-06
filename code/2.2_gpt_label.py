# curl -fsSL https://ollama.com/install.sh | sh
# pip install datasets ollama huggingface_hub

import os
import json
from datasets import load_dataset
from pydantic import BaseModel
from typing import Literal
from huggingface_hub import login
from dotenv import load_dotenv
import click
import tiktoken
from openai import OpenAI, OpenAIError
from sector_dict import SECTORS


load_dotenv()
global CLIENT
CLIENT = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

global MODEL
MODEL = "gpt-4o-mini"

global SYSTEM_PROMPT
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


def estimate_tokens(json_messages):
    tokenizer = tiktoken.encoding_for_model(MODEL)
    return len(tokenizer.encode(json.dumps(json_messages)))


def warn_user_about_tokens(batches):
    input_token_cost = 0.15
    output_token_cost = 0.60
    token_cost_per = 1000000
    input_token_count = 0
    output_token_count = 0
    for batch in batches:
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": batch}
        ]
        input_token_estimate = estimate_tokens(message)
        input_token_count += input_token_estimate
        output_token_count += input_token_estimate
    total_cost = ((input_token_count / token_cost_per) * input_token_cost) + ((output_token_count / token_cost_per) * output_token_cost)
    return click.confirm(
        "This will use about {} input tokens, {} output tokens, and cost about ${} to run. Do you want to continue?".format(
        input_token_count, round(output_token_count), round(total_cost, 2)
    )
    , default=False)


def gpt_label(example):
    try:
        completion = CLIENT.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(SECTORS[str(example['sector_code'])])},
                {"role": "user", "content": example['text']},
            ],
            temperature=0.2,
            response_format=ThoughtfulClassification,
        )
        response = completion.choices[0].message.parsed
    except OpenAIError as e:
        # Handle all OpenAI API errors
        print(f"Error: {e}")
        response = {'thoughts': f"Error: {e}", 'classifications': []}.items()
    for response_key, response_value in response:
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

    # Load data
    dataset = load_dataset('alex-miller/crs-2014-2023-housing-selection', split='train')
    unique_sectors = [str(sector) for sector in list(set(dataset['sector_code']))]
    missing_sectors = [sector for sector in unique_sectors if not sector in SECTORS]
    if len(missing_sectors) > 0:
        raise Exception(f"Please add the following sector codes to code/sector_dict.py:\n{"\n".join(missing_sectors)}")

    # Label
    if warn_user_about_tokens(dataset['text']) == True:
        dataset = dataset.map(gpt_label)
        dataset.push_to_hub('alex-miller/crs-2014-2023-housing-labeled-gpt')


if __name__ == '__main__':
    main()
