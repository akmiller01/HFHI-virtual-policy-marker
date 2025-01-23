import os
import shutil
from glob import glob
from openai import OpenAI
from dotenv import load_dotenv
import click
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
from openai_function_tokens import estimate_tokens

global OUT_FOLDER
# OUT_FOLDER = 'large_input/gpt_batch_files/crs_2014_2023'
OUT_FOLDER = 'large_input/gpt_batch_files/crs_2014_2023_market'

load_dotenv()
global CLIENT
CLIENT = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

global MODEL
MODEL = "gpt-4o-mini"

global BATCH_SIZE
BATCH_SIZE = 20000

global SYSTEM_PROMPT
global FUNCTIONS
SYSTEM_PROMPT = (
        "You are a helpful assistant that uses the extract_attributes function "
        "to extract attributes from user text as JSON. "
        "Always use the extract_attributes function."
        )
FUNCTIONS = [
    {
        "name": "extract_attributes",
        "description": "Add the attributes of user text to the database.",
        "parameters": {
            "type": "object",
                "properties": {
                    "housing_general": {
                        "type": "boolean",
                        "description": "The user text describes housing activities in general."
                    },
                    "homelessness_support": {
                        "type": "boolean",
                        "description": "The user text describes support for homelessness, including tents, encampments, and homeless shelters."
                    },
                    "transitional_housing": {
                        "type": "boolean",
                        "description": "The user text describes transitional housing, including emergency and refugee shelters and camps and semi-permanent supportive housing."
                    },
                    "incremental_housing": {
                        "type": "boolean",
                        "description": "The user text describes incremental housing, including housing sites, services and technical assistance, slum upgrading and structural repairs, and neighborhood integration."
                    },
                    "social_housing": {
                        "type": "boolean",
                        "description": "The user text describes social housing, including community land trusts, cooperative housing, and public housing."
                    },
                    "market_rent_own_housing": {
                        "type": "boolean",
                        "description": "The user text describes market-based housing solutions or rent-to-own policies, including social and subsidized rental, supported homeownership (first-time, rent-to-own), and market-rate affordable housing."
                    },
                    "urban": {
                        "type": "boolean",
                        "description": "The user text describes activities in urban settings in particular."
                    },
                    "rural": {
                        "type": "boolean",
                        "description": "The user text describes activities in rural settings in particular."
                    },
                    "climate_adaptation": {
                        "type": "boolean",
                        "description": "The user text describes activities relating to climate adaptation (responding to the effects of climate change or making communities more resilient to climate change)."
                    },
                    "climate_mitigation": {
                        "type": "boolean",
                        "description": "The user text describes activities relating to climate mitigation (reducing emissions or other causes of climate change)."
                    },
                },
            "required": [
                "housing_general",
                "homelessness_support",
                "transitional_housing",
                "incremental_housing",
                "social_housing",
                "market_rent_own",
                "urban",
                "rural",
                "climate_adaptation",
                "climate_mitigation"
                ],
        },
    }
]


def warn_user_about_tokens(batches):
    input_token_cost = 0.075
    output_token_cost = 0.3
    token_cost_per = 1000000
    input_token_count = 0
    output_token_count = 0
    fixed_output_len = 75
    for batch in batches:
        message = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Please extract the attributes from the following user text: {}".format(
                            batch
                        )
                    }
                ]
        input_token_estimate = estimate_tokens(message, functions=FUNCTIONS, function_call=None)
        input_token_count += input_token_estimate
        output_token_count += fixed_output_len
    total_cost = ((input_token_count / token_cost_per) * input_token_cost) + ((output_token_count / token_cost_per) * output_token_cost)
    return click.confirm(
        "This will use about {} input tokens, {} output tokens, and cost about ${} to run. Do you want to continue?".format(
        input_token_count, round(output_token_count), round(total_cost, 2)
    )
    , default=False)


def create_batch_files(batch):
    batch_index = 0
    batch_json_file = os.path.join(OUT_FOLDER, f'batch-{batch_index}.jsonl')
    while os.path.exists(batch_json_file):
        batch_index += 1
        batch_json_file = os.path.join(OUT_FOLDER, f'batch-{batch_index}.jsonl')
    batch_csv_file = os.path.join(OUT_FOLDER, f'batch-{batch_index}.csv')
    requests_list = list()
    for i, id in enumerate(batch['id']):
        text = batch['text'][i]
        request_obj = {
            'custom_id': f'crs-{id}',
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': MODEL,
                'functions': FUNCTIONS,
                'messages': [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Please extract the attributes from the following user text: {}".format(
                            text
                        )
                    }
                ]
            }
        }
        request_obj_json = json.dumps(request_obj, ensure_ascii=False).encode('utf-8')
        requests_list.append(request_obj_json)
    with open(batch_json_file, 'wb') as json_file:
        for request in requests_list:
            json_file.write(request)
            json_file.write(b'\n')
    Dataset.from_dict(batch).to_csv(batch_csv_file)


def main():
    # Load data
    # dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    # dataset = dataset.add_column('id', range(dataset.num_rows))

    # Redefine market_rent_own
    dataset = load_dataset("csv", data_files="large_input/crs_2014_2023_gpt_batched.csv", split="train")
    dataset = dataset.filter(lambda example: example['market_rent_own'] == 1)
    cols_to_remove = dataset.column_names
    cols_to_keep = ['id', 'text', 'sector_code']
    cols_to_remove = [col for col in cols_to_remove if col not in cols_to_keep]
    dataset = dataset.remove_columns(cols_to_remove)

    if warn_user_about_tokens(dataset['text']) == True:
        # Delete previous run
        shutil.rmtree(OUT_FOLDER)
        os.makedirs(OUT_FOLDER, exist_ok=True)

        # Create batches
        dataset.map(create_batch_files, batched=True, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main()
