from huggingface_hub import login
from datasets import load_dataset
import os
import pandas as pd
from tqdm import tqdm


def create_unique_text(row):
    title = row['project_title'] if not pd.isna(row['project_title']) else ''
    short = row['short_description'] if not pd.isna(row['short_description']) else ''
    long = row['long_description'] if not pd.isna(row['long_description']) else ''

    project_text = long
    if short.lower() not in project_text.lower():
        project_text = '{} {}'.format(short, project_text)

    if title.lower() not in project_text.lower():
        project_text = '{} {}'.format(title, project_text)

    return project_text.strip()

def main():
    tqdm.pandas()  # Enable tqdm for Pandas
    df = pd.read_csv('./large_input/crs_2014_2023.csv')

    # Create text column
    df['text'] = df.progress_apply(create_unique_text, axis=1)

    # Load labeled texts from HF
    labels = load_dataset('alex-miller/crs-2014-2023-housing-labeled-phi4', split='train')
    labels = labels.remove_columns(['sector_code'])
    labels = labels.to_pandas()

    df = df.merge(labels, how="left", on="text")
    df.to_csv("large_output/crs_2014_2023_phi4_labeled.csv")



if __name__ == '__main__':
    main()
