import os
import json
from dotenv import load_dotenv
import click
import tiktoken
from openai import OpenAI, OpenAIError
import pandas as pd
from tqdm import tqdm
from wb_definitions import SUFFIX, RETRIEVAL_TOPICS


SYSTEM_PROMPT = """
### Task

You are extracting and creating keywords for use in **literal text search**.

Given:
- A **concept**
- An **input phrase**

Your goal is to extract and create a **small set of keywords** that:
- Represent the **core meaning** of the input phrase
- Are **highly specific** to the given concept

### Keyword Rules

1. Extract single-word keywords from the input phrase that represent the phrase and are not general or ambiguous.
2. Include other related words that reflect the input phrase in the specific context of the given concept.
3. Only include terms that, **on their own**, clearly and specifically relate to the given concept.
4. **Do not include general or ambiguous words**, such as: `provision`, `support`, `assistance`, `supply`, `program`, `help`, `services`, etc.
5. A keyword must be **clearly associated with the concept** and unlikely to cause false positives in other domains (e.g., food, education, health).

### Output Format (STRICT)

For each keyword:
- Provide translations in **English, Spanish, French, German, Italian, Polish, Portuguese**
- Separate all translations for a keyword with a **single pipe character (`|`)**.
- Separate multiple keywords with a **single pipe character (`|`)** — **not spaces**, **not commas**, **not newlines**.
- Do not include any explanations or extra text — return only the pipe separated keywords.
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


def gpt_keywords(concept, input_phrase):
    try:
        response = CLIENT.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=f"Concept: {concept}\nInput phrase: {input_phrase}",
            service_tier="flex"
        )
        response_text = response.output_text
    except OpenAIError as e:
        # Handle all OpenAI API errors
        print(f"Error: {e}")
        response_text = ""
    keywords = response_text.split("|")

    return keywords


def main():
    simulated_batches = list()
    for concept in RETRIEVAL_TOPICS.keys():
        input_phrases = RETRIEVAL_TOPICS[concept]
        for input_phrase in input_phrases:
            simulated_batches.append(input_phrase)
    
    if warn_user_about_tokens(simulated_batches):
        unique_keywords = list()
        results = list()
        with tqdm(total=len(simulated_batches)) as pbar:
            for concept in RETRIEVAL_TOPICS.keys():
                input_phrases = RETRIEVAL_TOPICS[concept]
                for input_phrase in input_phrases:
                    pbar.update(1)
                    unique_keywords.append(input_phrase.lower().strip())
                    results_dict = {
                        'word': input_phrase.lower().strip(),
                        'input_phrase': input_phrase,
                        'concept': concept
                    }
                    results.append(results_dict)
                    output_keywords = gpt_keywords(concept, input_phrase)
                    for keyword in output_keywords:
                        if keyword not in unique_keywords:
                            lower_keyword = keyword.lower().strip()
                            unique_keywords.append(lower_keyword)
                            results_dict = {
                                'word': lower_keyword,
                                'input_phrase': input_phrase,
                                'concept': concept
                            }
                            results.append(results_dict)
        results_df = pd.DataFrame.from_records(results)
        results_df.to_csv(f"input/keywords{SUFFIX}.csv", index=False)



if __name__ == '__main__':
    main()
