# curl -fsSL https://ollama.com/install.sh | sh
# pip install datasets ollama

import json
from datasets import load_dataset
import ollama
from ollama import chat
from ollama import ChatResponse

global MODEL
global REFRESH_MODELS
MODEL = "mistral"

global SYSTEM_PROMPT
global DEFINITIONS
global FORMAT
SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies text. "
    "If the text explicitly describes {}, then the answer is true, otherwise the answer is false. "
    "Base your response only on the text and no other external context. "
    "You respond in JSON format, first giving thoughts in as few words as needed about whether the text matches the definition above in the 'thoughts' key and "
    "then giving your answer in the 'answer' key."
    )
DEFINITIONS = {
    "housing": "any activities relating to housing, except for project staff housing",
    "homelessness": "tents for the homeless, encampments for the homeless, or homeless shelters",
    "transitional": "emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
    "incremental": "housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration",
    "social": "community land trusts, cooperative housing, or public housing",
    "market": "subsidized home-rental, first-time home buyer programs, rent-to-own housing, or market-rate housing",
    "urban": "activities in specific urban locations",
    "rural": "activities in specific rural locations",
    "adaptation": "adapting to the direct effects of climate change",
    "mitigation": "mitigating the direct causes of climate change"
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

    return example


def main():
    # Pull model
    ollama.pull(MODEL)

    # Load data
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    dataset = dataset.add_column('id', range(dataset.num_rows))
    dataset = dataset.filter(lambda example: example['sector_code'] in [16030, 16040])
    dataset = dataset.shuffle(seed=1337).select(range(10))

    # Label
    dataset = dataset.map(ollama_label)
    dataset.to_csv('large_input/ollama_test.csv')


if __name__ == '__main__':
    main()
