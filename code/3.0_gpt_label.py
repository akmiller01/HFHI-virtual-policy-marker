import os
import json
import time
import random
import pandas as pd
from dotenv import load_dotenv
import click
import tiktoken
from openai import OpenAI
from hfhi_definitions import SUFFIX, SYSTEM_PROMPT, DEFINITIONS, ReasonedClassification
# from wb_definitions import SUFFIX, SYSTEM_PROMPT, DEFINITIONS, ReasonedClassification
from common import SECTORS
from tqdm import tqdm


load_dotenv()
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=900.0)
MODEL = "gpt-5-nano"
# Temporary debug flag: when True, limit labeling to first N rows
DEBUG = True
DEBUG_LIMIT = 10


def estimate_tokens_for_message(text):
    tokenizer = tiktoken.encoding_for_model(MODEL)
    message = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    return len(tokenizer.encode(json.dumps(message)))


def warn_user_about_tokens(texts):
    input_token_cost = 0.025
    output_token_cost = 0.20
    token_cost_per = 1000000
    input_token_count = 0
    output_token_count = 0
    for text in texts:
        input_token_estimate = estimate_tokens_for_message(text)
        input_token_count += input_token_estimate
        output_token_count += input_token_estimate
    total_cost = ((input_token_count / token_cost_per) * input_token_cost) + ((output_token_count / token_cost_per) * output_token_cost)
    return click.confirm(
        "This will use about {} input tokens, {} output tokens, and cost about ${} to run. Do you want to continue?".format(
            input_token_count, round(output_token_count), round(total_cost, 2)
        ),
        default=False,
    )


def process_parsed_response(parsed):
    # Normalize parsed output (supports dicts, iterables, and pydantic BaseModel-like objects)
    if parsed is None:
        return {}

    # If it's already a dict, use it directly
    if isinstance(parsed, dict):
        data = parsed
    else:
        # pydantic v1 uses .dict(), v2 uses .model_dump(); prefer model_dump if available
        if hasattr(parsed, "model_dump") and callable(getattr(parsed, "model_dump")):
            try:
                data = parsed.model_dump()
            except Exception:
                data = None
        elif hasattr(parsed, "dict") and callable(getattr(parsed, "dict")):
            try:
                data = parsed.dict()
            except Exception:
                data = None
        else:
            # Fallback: try to coerce to dict
            try:
                data = dict(parsed)
            except Exception:
                data = None

    if not data:
        return {}

    out = {}
    for response_key, response_value in data.items():
        if isinstance(response_value, list):
            for definition_key in DEFINITIONS.keys():
                out[definition_key] = definition_key in response_value
        else:
            out[response_key] = response_value
    return out


def gpt_label_with_retry(text, purpose_code, max_retries=5, base_backoff=1.0):
    sector_label = None
    try:
        sector_label = SECTORS.get(str(purpose_code), '')
    except Exception:
        sector_label = ''

    user_content = f"{text}\nSector: {sector_label}"

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = CLIENT.beta.chat.completions.parse(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
                response_format=ReasonedClassification,
                service_tier="flex",
            )
            parsed = completion.choices[0].message.parsed
            return process_parsed_response(parsed)
        except Exception as e:
            last_exc = e
            wait = min(60, base_backoff * (2 ** (attempt - 1)))
            wait = wait + random.uniform(0, 1)
            print(f"Attempt {attempt} failed: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    # All retries exhausted — raise exception to let caller handle it
    raise last_exc


def main(input_path=None, output_path=None):
    # Infer paths
    if input_path is None:
        # Default to the preprocessed file produced by 2.0_preprocess_crs.py
        input_path = os.path.join('large_input', f'crs_2024_update_preprocessed.csv')
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join('large_input', f'{base}_labeled{SUFFIX}.csv')

    # Load preprocessed CSV
    df = pd.read_csv(input_path, dtype={'PurposeCode': object})
    # Ensure an original index column to resume reliably
    if '_orig_index' not in df.columns:
        df['_orig_index'] = df.index

    # If debugging, limit the dataframe early so all downstream steps use the subset.
    # Only keep rows whose PurposeCode is 16030 or 16040, up to DEBUG_LIMIT.
    if DEBUG:
        print(f"DEBUG mode enabled — limiting to first {DEBUG_LIMIT} rows with purpose code 16030 or 16040")
        mask = df['PurposeCode'].astype(str).str.strip().isin(['16030', '16040'])
        filtered = df[mask]
        if filtered.empty:
            print(f"No rows found with purpose codes 16030/16040 — falling back to first {DEBUG_LIMIT} rows")
            df = df.head(DEBUG_LIMIT).copy()
        else:
            df = filtered.head(DEBUG_LIMIT).copy()

    # Prepare list of texts for token warning
    texts = df['text'].astype(str).tolist()
    if not warn_user_about_tokens(texts):
        print('User declined to proceed based on token/cost estimate.')
        return

    # Load existing results if present to resume
    processed_indices = set()
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path, dtype={'_orig_index': int})
        if '_orig_index' in existing.columns:
            processed_indices = set(existing['_orig_index'].astype(int).tolist())
        else:
            # No index column — assume nothing processed
            processed_indices = set()

    # Iterate and process one-by-one, appending results
    first_write = not os.path.exists(output_path)
    rows_written = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Labeling'):
        idx = int(row['_orig_index'])
        if idx in processed_indices:
            continue

        try:
            parsed_out = gpt_label_with_retry(row['text'], row.get('PurposeCode', None))
        except Exception as e:
            print(f"Exhausted retries for index {idx}: {e}")
            raise

        # Build output record
        out_rec = {
            '_orig_index': idx,
            'text': row['text'],
            'PurposeCode': row.get('PurposeCode', None),
        }
        out_rec.update(parsed_out)

        out_df = pd.DataFrame([out_rec])
        out_df.to_csv(output_path, mode='a', header=first_write, index=False)
        first_write = False
        rows_written += 1

    print(f'Done — wrote {rows_written} new rows to {output_path}')


if __name__ == '__main__':
    main()
