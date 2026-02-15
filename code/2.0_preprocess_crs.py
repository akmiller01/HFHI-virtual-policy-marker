import os
import pandas as pd
from tqdm import tqdm
import tiktoken


MODEL = "gpt-5-nano"
tokenizer = tiktoken.encoding_for_model(MODEL)


def create_unique_text(row):
    title = row['ProjectTitle'] if not pd.isna(row['ProjectTitle']) else ''
    short = row['ShortDescription'] if not pd.isna(row['ShortDescription']) else ''
    long = row['LongDescription'] if not pd.isna(row['LongDescription']) else ''

    project_text = long or ''
    if short and short.lower() not in project_text.lower():
        project_text = '{} {}'.format(short, project_text)

    if title and title.lower() not in project_text.lower():
        project_text = '{} {}'.format(title, project_text)

    return project_text.strip()


def estimate_token_counts(texts):
    return sum(len(tokenizer.encode(t)) for t in texts)


def estimate_cost_from_tokens(total_tokens):
    input_token_cost = 0.025
    output_token_cost = 0.20
    token_cost_per = 1000000
    input_token_count = total_tokens
    output_token_count = total_tokens
    total_cost = ((input_token_count / token_cost_per) * input_token_cost) + ((output_token_count / token_cost_per) * output_token_cost)
    return input_token_count, output_token_count, total_cost


def main(input_path='./large_input/crs_2024_update.csv', output_path=None):
    tqdm.pandas()

    if output_path is None:
        base, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(os.path.dirname(input_path), f"{base}_preprocessed{ext}")

    usecols = ['ProjectTitle', 'ShortDescription', 'LongDescription', 'PurposeCode']
    try:
        df = pd.read_csv(input_path, usecols=usecols, dtype={'ProjectTitle': str, 'ShortDescription': str, 'LongDescription': str, 'PurposeCode': object})
    except Exception:
        # Fallback: load full CSV and attempt to proceed
        df = pd.read_csv(input_path, dtype=str)

    starting_rows = df.shape[0]
    print('Starting rows:', starting_rows)

    # Create text column
    df['text'] = df.progress_apply(create_unique_text, axis=1)
    if 'PurposeCode' not in df.columns:
        df['PurposeCode'] = None
    df = df[['text', 'PurposeCode']]

    # De-duplicate
    prededup_rows = df.shape[0]
    df = df.drop_duplicates(subset=['text'])
    postdedup_rows = df.shape[0]
    print('Rows removed by de-duplication:', prededup_rows - postdedup_rows)

    # Remove blanks
    preblank_rows = df.shape[0]
    df = df[df['text'].str.strip().astype(bool)]
    postblank_rows = df.shape[0]
    print('Rows removed as blank:', preblank_rows - postblank_rows)
    print('Ending rows:', postblank_rows)

    # Count tokens and estimate cost
    total_token_count = estimate_token_counts(df['text'].astype(str).tolist())
    input_tokens, output_tokens, est_cost = estimate_cost_from_tokens(total_token_count)
    print(f'Total token count: {total_token_count}')
    print(f'Estimated input tokens: {input_tokens}, estimated output tokens: {output_tokens}, estimated cost: ${est_cost:.2f}')

    # Write CSV
    df.to_csv(output_path, index=False)
    print('Wrote preprocessed CSV to:', output_path)


if __name__ == '__main__':
    main()
