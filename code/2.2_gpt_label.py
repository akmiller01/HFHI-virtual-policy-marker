import os
import json
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import click
import tiktoken
from openai import OpenAI, OpenAIError
from model_common import SYSTEM_PROMPT, DEFINITIONS, ThoughtfulClassification


load_dotenv()
global CLIENT
CLIENT = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)


global MODEL
MODEL = "gpt-4o-mini"


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
                {"role": "system", "content": SYSTEM_PROMPT},
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

    # Label
    if warn_user_about_tokens(dataset['text']) == True:
        dataset = dataset.map(gpt_label)
        dataset.push_to_hub('alex-miller/crs-2014-2023-housing-labeled-gpt')


if __name__ == '__main__':
    main()
