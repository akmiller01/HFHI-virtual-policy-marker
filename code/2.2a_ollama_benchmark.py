# curl -fsSL https://ollama.com/install.sh | sh
# pip install datasets ollama huggingface_hub python-dotenv pydantic tqdm

import json
import pandas as pd
import ollama
from ollama import chat
from ollama import ChatResponse
from hfhi_definitions import SYSTEM_PROMPT, SYSTEM_PROMPT_V2, DEFINITIONS, DEFINITIONS_V2, ReasonedClassification, ReasonedClassificationV2
from common import SECTORS
from util_self_termination import main as self_terminate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


global MODEL
MODEL = "phi4"


def ollama_label(example):
    response: ChatResponse = chat(
        model=MODEL,
        format=ReasonedClassification.model_json_schema(),
        messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT,
            },
            {
                'role': 'user',
                'content': '{}\nSector: {}'.format(
                    example['text'],
                    SECTORS[str(example['PurposeCode'])],
                ),
            },
        ]
    )
    parsed_response_content = json.loads(response.message.content)
    for response_key in parsed_response_content:
        response_value = parsed_response_content[response_key]
        if type(response_value) is list:
            for definition_key in DEFINITIONS.keys():
                example[definition_key] = definition_key in response_value
        else:
            example[response_key] = response_value

    return example


def get_predictions(example, prompt, definitions, ReasonedCls):
    response: ChatResponse = chat(
        model=MODEL,
        format=ReasonedCls.model_json_schema(),
        messages=[
            {
                'role': 'system',
                'content': prompt,
            },
            {
                'role': 'user',
                'content': '{}\nSector: {}'.format(
                    example['text'],
                    SECTORS[str(example['PurposeCode'])],
                ),
            },
        ]
    )
    parsed_response_content = json.loads(response.message.content)
    pred = {}
    for response_key in parsed_response_content:
        response_value = parsed_response_content[response_key]
        if type(response_value) is list:
            for definition_key in definitions.keys():
                pred[definition_key] = definition_key in response_value
    return pred


def evaluate(y_true, y_pred, definition_keys):
    results = {}
    for key in definition_keys:
        y_true_col = [row[key] for row in y_true]
        y_pred_col = [row[key] for row in y_pred]
        acc = accuracy_score(y_true_col, y_pred_col)
        prec, recall, f1, _ = precision_recall_fscore_support(y_true_col, y_pred_col, average='binary', zero_division=0)
        results[key] = {
            'accuracy': acc,
            'precision': prec,
            'recall': recall,
            'f1': f1
        }
    return results


def main():
    # Pull model
    ollama.pull(MODEL)

    # Load benchmark CSV
    df = pd.read_csv('input/benchmark_hfhi.csv')
    definition_keys_v1 = list(DEFINITIONS.keys())
    definition_keys_v2 = list(DEFINITIONS_V2.keys())

    # Prepare ground truth
    y_true_v1 = df[definition_keys_v1].astype(bool).to_dict(orient='records')
    y_true_v2 = df[definition_keys_v2].astype(bool).to_dict(orient='records')

    # Run predictions for V1
    y_pred_v1 = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='V1 Benchmark'):
        example = {'text': row['text'], 'PurposeCode': row['PurposeCode']}
        pred = get_predictions(example, SYSTEM_PROMPT, DEFINITIONS, ReasonedClassification)
        y_pred_v1.append(pred)

    # Run predictions for V2
    y_pred_v2 = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='V2 Benchmark'):
        example = {'text': row['text'], 'PurposeCode': row['PurposeCode']}
        pred = get_predictions(example, SYSTEM_PROMPT_V2, DEFINITIONS_V2, ReasonedClassificationV2)
        y_pred_v2.append(pred)

    # Evaluate
    print('V1 Results:')
    results_v1 = evaluate(y_true_v1, y_pred_v1, definition_keys_v1)
    for key, metrics in results_v1.items():
        print(f'{key}: {metrics}')

    print('\nV2 Results:')
    results_v2 = evaluate(y_true_v2, y_pred_v2, definition_keys_v2)
    for key, metrics in results_v2.items():
        print(f'{key}: {metrics}')

    self_terminate()


if __name__ == '__main__':
    main()
