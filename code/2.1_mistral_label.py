# curl -fsSL https://ollama.com/install.sh | sh

import json
from datasets import load_dataset
import ollama
from ollama import chat
from ollama import ChatResponse

global MODEL
global REFRESH_MODELS
MODEL = "mistral"
REFRESH_MODELS = False

global SYSTEM_PROMPT
global DEFINITIONS
global FORMAT
SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies text. "
    "If the text explicitly describes {}, then the answer is true, otherwise the answer is false. "
    "Base your response only on the text and no other external context. "
    "You respond in JSON format, first giving your thoughts about whether the text matches the definition above in the 'thoughts' key, "
    "then giving your answer in the 'answer' key, and finally rating whether your confidence about your answer is high in the 'confidence_high' key."
    )
DEFINITIONS = {
    "housing": "providing housing to people",
    "homelessness": "tents for the homeless, encampments for the homeless, or homeless shelters",
    "transitional": "emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
    "incremental": "housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration",
    "social": "community land trusts, cooperative housing, or public housing",
    "market": "subsidized housing rental, first-time homebuyer programs, rent-to-own housing, or market-rate housing",
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
        },
        "confidence_high": {
            "type": "boolean"
        }
    },
    "required": [
        "thoughts",
        "answer",
        "confidence_high"
    ]
}


def create_models():
    current_models = ollama.list().models
    current_model_names = [model.model.split(":")[0] for model in current_models]
    ollama.pull(MODEL)
    for key in DEFINITIONS.keys():
        if key in current_model_names and not REFRESH_MODELS:
            print(f"{key} model already exists, skipping creation...")
            continue

        print(f"Creating {key} model...")
        definition = DEFINITIONS[key]
        definition_system_prompt = SYSTEM_PROMPT.format(definition)
        ollama.create(
            model=key, 
            from_=MODEL, 
            system=definition_system_prompt
        )

def mistral_label(example):
    for key in DEFINITIONS.keys():
        response: ChatResponse = chat(
            model=key,
            format=FORMAT,
            messages=[
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
    # Create models
    create_models()

    # Load data
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    dataset = dataset.add_column('id', range(dataset.num_rows))
    dataset = dataset.filter(lambda example: example['sector_code'] in [16030, 16040])
    dataset = dataset.shuffle(seed=1337).select(range(10))

    # Label
    dataset = dataset.map(mistral_label)
    dataset.to_csv('large_input/mistral_test.csv')


if __name__ == '__main__':
    main()
