# curl -fsSL https://ollama.com/install.sh | sh
# pip install datasets ollama

# TODO:
# 1. Look into UNHABITAT trigger

import json
from datasets import load_dataset, concatenate_datasets
import ollama
from ollama import chat
from ollama import ChatResponse

global MODEL
global REFRESH_MODELS
MODEL = "mistral"

global SYSTEM_PROMPT
global URBAN_SYSTEM_PROMPT
global DEFINITIONS
global FORMAT
SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies text. "
    "You are classifying whether the text explicitly describes {}. "
    "Base your response only on the text and no other external context. "
    "If the text is too short or too vague to give a definitive answer, mark it as false. "
    "You respond in JSON format, first giving thoughts in as few words as needed about whether the text matches the definition above in the 'thoughts' key and "
    "then giving your answer in the 'answer' key."
)
URBAN_SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies text. "
    "You are classifying whether the text explicitly describes activities in specific urban locations, specific rural locations, both urban and rural locations, or neither."
    "Base your response only on the text and no other external context. "
    "If the text is too short or too vague to give a definitive answer, mark it as neither. "
    "You respond in JSON format, first giving thoughts in as few words as needed about whether the text matches the definitions above in the 'thoughts' key and "
    "then giving your answer in the 'answer' key. Possible answer choices are 'Urban', 'Rural', 'Both', or 'Neither'."
)
DEFINITIONS = {
    "housing": "any of the following: housing, housing policy, tents for the homeless, encampments for the homeless, homeless shelters, emergency shelters, refugee shelters, refugee camps, temporary supportive housing, housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, neighborhood integration, community land trusts, cooperative housing, public housing, subsidized home-rental, subsidized mortgages, rent-to-own housing, or market-rate housing",
    "homelessness": "tents for the homeless, encampments for the homeless, or homeless shelters",
    "transitional": "emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
    "incremental": "housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration",
    "social": "community land trusts, cooperative housing, or public housing",
    "market": "subsidized home-rental, subsidized mortgages, rent-to-own housing, or market-rate housing",
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
        else:
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
    # Pull model
    ollama.pull(MODEL)

    # Load data
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    dataset = dataset.add_column('id', range(dataset.num_rows))
    pos = dataset.filter(lambda example: example['sector_code'] in [16030, 16040]).shuffle(seed=1337).select(range(5))
    neg = dataset.filter(lambda example: example['sector_code'] not in [16030, 16040]).shuffle(seed=1337).select(range(5))
    dataset = concatenate_datasets([neg, pos])

    # Label
    dataset = dataset.map(ollama_label)
    dataset.to_csv('large_input/ollama_test.csv')


if __name__ == '__main__':
    main()
