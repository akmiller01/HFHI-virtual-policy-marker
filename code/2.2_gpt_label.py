import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import tiktoken
import click
import json
from datasets import load_dataset, concatenate_datasets


load_dotenv()
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

MODEL = "gpt-4o-mini"

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
                        "description": "The user text describes support for homelessness, including tents and encampments."
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
                    "market_rent_own": {
                        "type": "boolean",
                        "description": "The user text describes market rent and own policies, including social and subsidized rental, supported homeownership (first-time, rent-to-own), and market-rate affordable housing."
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


def warn_user_about_tokens(tokenizer, batches, other_prompts):
    token_cost = 0.15
    token_cost_per = 1000000
    token_count = 0
    for batch in batches:
        token_count += len(tokenizer.encode(batch))
        token_count += len(tokenizer.encode(other_prompts))
    return click.confirm(
        "This will use at least {} tokens and cost at least ${} to run. Do you want to continue?".format(
        token_count, round((token_count / token_cost_per) * token_cost, 2)
    )
    , default=False)


def gpt_inference(example):
    user_text = example['text']
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Please extract the attributes from the following user text: {}".format(
                user_text
            )
        }
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            functions=FUNCTIONS, messages=messages
        )
        function_args = json.loads(
            response.choices[0].message.function_call.arguments
        )
        for key in function_args:
            example[key] = function_args[key]
    except OpenAIError as e:
        print(f"Error: {e}")
    
    return example


def main():
    
    tokenizer = tiktoken.encoding_for_model(MODEL)
    
    # Load data
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    pos = dataset.filter(lambda row: row['sector_code'] in [16030, 16040])
    neg = dataset.filter(lambda row: row['sector_code'] not in [16030, 16040])
    neg = neg.shuffle(seed=1337).select(range(pos.num_rows))
    dataset = concatenate_datasets([pos, neg])

    if warn_user_about_tokens(tokenizer, dataset['text'], other_prompts=json.dumps(FUNCTIONS)) == True:
        # Inference
        dataset = dataset.map(gpt_inference, num_proc=4)
        dataset.to_csv("large_input/crs_2014_2023_gpt.csv")


if __name__ == '__main__':
    main()
