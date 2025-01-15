import os
import datetime
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import re


load_dotenv()
global CLIENT
CLIENT = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

def extract_percentage(input_string):
    # Search for a percentage pattern in the string
    match = re.search(r'(\d+)%', input_string)
    if match:
        # Convert the captured number to a decimal
        return int(match.group(1)) / 100
    return None  # Return None if no percentage is found


def main():
    records = list()
    batches = CLIENT.batches.list()
    for batch in batches:
        if batch.status == 'completed':
            record = {
                'created_at': datetime.datetime.fromtimestamp(batch.created_at),
                'in_progress_at': datetime.datetime.fromtimestamp(batch.in_progress_at),
                'finalizing_at': datetime.datetime.fromtimestamp(batch.finalizing_at),
                'completed_at': datetime.datetime.fromtimestamp(batch.completed_at),
                'requests': batch.request_counts.total
            }
            records.append(record)

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by='created_at')
    with open('output/gpt.log', 'r') as log_file:
        log_lines = log_file.read().split('\n')
    complete_and_percent_lines = [line for line in log_lines if ('%' in line or '!' in line) and 'Entire' not in line]
    extracted_percentages = [extract_percentage(line) for line in complete_and_percent_lines]
    
    # Align the extracted percentages with None values in the DataFrame
    aligned_list = []
    df_index = 0

    for value in extracted_percentages:
        if value is None:
            aligned_list.append(None)  # Keep None for the existing DataFrame rows
            df_index += 1  # Move to the next row in the DataFrame
        else:
            aligned_list.append(value)  # Add percentage values

    # Create a new DataFrame with aligned data
    expanded_df = pd.DataFrame({'extracted_percentages': aligned_list})

    # Add other columns from the original DataFrame to the expanded DataFrame
    expanded_df.loc[expanded_df['extracted_percentages'].isna(), ['created_at', 'in_progress_at', 'finalizing_at', 'completed_at', 'requests']] = df.values

    expanded_df.to_csv('output/gpt_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
