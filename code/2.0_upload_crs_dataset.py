from huggingface_hub import login
from datasets import Dataset
from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm
import tiktoken


global tokenizer
tokenizer = tiktoken.encoding_for_model('gpt-4o-mini')


def count_tokens(example):
    token_count = 0
    if example['text'] is not None and len(example['text']) > 0:
        token_count = len(tokenizer.encode(example['text']))
    return {'count': token_count}


def create_unique_text(row):
    title = row['ProjectTitle'] if not pd.isna(row['ProjectTitle']) else ''
    short = row['ShortDescription'] if not pd.isna(row['ShortDescription']) else ''
    long = row['LongDescription'] if not pd.isna(row['LongDescription']) else ''

    project_text = long
    if short.lower() not in project_text.lower():
        project_text = '{} {}'.format(short, project_text)

    if title.lower() not in project_text.lower():
        project_text = '{} {}'.format(title, project_text)

    return project_text.strip()

def main():
    tqdm.pandas()  # Enable tqdm for Pandas
    df = pd.read_csv(
        './large_input/crs_2014_2023.csv',
        usecols=['ProjectTitle', 'ShortDescription', 'LongDescription', 'PurposeCode'],
        dtype={
            'ProjectTitle': str, 'ShortDescription': str, 'LongDescription': str, 'PurposeCode': int
        }
    )
    starting_rows = df.shape[0]
    print('Starting rows:', starting_rows)

    # Create text column
    df['text'] = df.progress_apply(create_unique_text, axis=1)
    df = df[['text', 'PurposeCode']]
   
    # De-duplicate
    prededup_rows = df.shape[0]
    df = df.drop_duplicates(subset=['text'])
    postdedup_rows = df.shape[0]
    print('Rows removed by de-duplication:', prededup_rows - postdedup_rows)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    # Remove blanks
    preblank_rows = dataset.num_rows
    dataset = dataset.filter(lambda example: example['text'] != '', num_proc=8)
    postblank_rows = dataset.num_rows
    print('Rows removed as blank:', preblank_rows - postblank_rows)
    print('Ending rows:', postblank_rows)

    # Count tokens
    total_token_count = sum(dataset.map(count_tokens)['count'])
    print('Total token count: {}'.format(total_token_count))

    # Push to HF
    dataset.push_to_hub('alex-miller/crs-2014-2023')


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()
