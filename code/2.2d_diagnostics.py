import os
import datetime
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd


load_dotenv()
global CLIENT
CLIENT = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)


records = list()


def main():
    batches = CLIENT.batches.list()
    for batch in batches:
        if batch.status == 'completed':
            record = {
                'created_at': datetime.datetime.fromtimestamp(batch.created_at),
                'in_progress_at': datetime.datetime.fromtimestamp(batch.in_progress_at),
                'finalizing_at': datetime.datetime.fromtimestamp(batch.finalizing_at),
                'completed_at': datetime.datetime.fromtimestamp(batch.completed_at)
            }
            records.append(record)

    df = pd.DataFrame.from_records(records)
    df.to_csv('output/gpt_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
