import os
import json
from datasets import load_dataset
import ollama
from ollama import chat
from ollama import ChatResponse
from model_common import SYSTEM_PROMPT, DEFINITIONS, ThoughtfulClassification


global MODEL
MODEL = "phi4"


def ollama_label(example):
    response: ChatResponse = chat(
        model=MODEL,
        format=ThoughtfulClassification.model_json_schema(),
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT,
            },
            {
                'role': 'user',
                'content': example['text'],
            },
        ],
        # options={'temperature': 0.2}
    )
    parsed_response_content = json.loads(response.message.content)
    for response_key in parsed_response_content:
        response_value = parsed_response_content[response_key]
        if type(response_value) is list:
            for definition_key in DEFINITIONS.keys():
                example[definition_key + ' AI'] = definition_key in response_value
        else:
            example[response_key + ' AI'] = response_value

    return example


def main():
    # Pull model
    ollama.pull(MODEL)

    # Load data
    dataset = load_dataset('csv', data_files='input/manually_coded_for_accuracy.csv', split='train')
    dataset = dataset.remove_columns(
        ['thoughts AI', 'Housing AI', 'Homelessness AI',
         'Transitional AI', 'Incremental AI', 'Social AI',
         'Market AI', 'Urban AI', 'Rural AI', 'DK notes',
         'DK key note', 'Selection type']
    )

    # Label
    dataset = dataset.map(ollama_label)
    dataset.to_csv("input/accuracy_20250207_nosector2.csv")


if __name__ == '__main__':
    main()
