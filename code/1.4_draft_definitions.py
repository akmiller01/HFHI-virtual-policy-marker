import os
import json
from dotenv import load_dotenv
import click
import tiktoken
from openai import OpenAI, OpenAIError
from tqdm import tqdm
from wb_definitions import SUFFIX, RETRIEVAL_TOPICS


SYSTEM_PROMPT = """
You are writing a classification instruction for a specific subsector within the housing sector, in the context of international assistance.

The user input will consist of:
- A subsector name.
- A bulleted list of concepts or keywords related to that subsector.

Your task is to write a **single, well-structured paragraph** that:
- Begins with this format: `{Subsector name}: when the text explicitly describes ...`.
- Provides a **clear and concise definition** of the subsector.
- Highlights **key activities, concepts, or topics** that fall within the subsector.

Guidelines:
- Use **short, clear sentences**.
- Do **not** use bullet points or newlines.
- Avoid run-on sentences or excessive clause stacking.
- Use formal, neutral, and instructional language suitable for text classification guidance.

Return only the paragraph. Do not include explanations, formatting symbols, or any other text.
"""


load_dotenv()
global CLIENT
CLIENT = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    timeout=900.0
)


global MODEL
global TOKENIZER_MODEL
MODEL = "o4-mini"
TOKENIZER_MODEL = "gpt-4o"



def estimate_tokens(json_messages):
    tokenizer = tiktoken.encoding_for_model(TOKENIZER_MODEL)
    return len(tokenizer.encode(json.dumps(json_messages)))


def warn_user_about_tokens(batches):
    input_token_cost = 0.55
    output_token_cost = 2.20
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
        output_token_count += 4000
    total_cost = ((input_token_count / token_cost_per) * input_token_cost) + ((output_token_count / token_cost_per) * output_token_cost)
    return click.confirm(
        "This will use about {} input tokens, {} output tokens, and cost about ${} to run. Do you want to continue?".format(
        input_token_count, round(output_token_count), round(total_cost, 2)
    )
    , default=False)


def gpt_definition(input):
    try:
        response = CLIENT.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=input,
            service_tier="flex"
        )
        response_text = response.output_text
    except OpenAIError as e:
        # Handle all OpenAI API errors
        print(f"Error: {e}")
        response_text = ""
    return response_text


def main():
    simulated_batches = list(RETRIEVAL_TOPICS.keys())

    results = list()
    if warn_user_about_tokens(simulated_batches):
        with tqdm(total=len(simulated_batches)) as pbar:
            for concept in RETRIEVAL_TOPICS.keys():
                pbar.update(1)
                input = f"Subsector name: {concept}\n {'\n- '.join(RETRIEVAL_TOPICS[concept])}"
                output = gpt_definition(input)
                results.append(output)
                
        with open(f"input/definitions{SUFFIX}.txt", "w") as txt_file:
            txt_file.write("\n".join(results))


if __name__ == '__main__':
    main()
