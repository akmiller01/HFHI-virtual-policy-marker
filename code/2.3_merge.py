from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from hfhi_definitions import SUFFIX


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
    df = pd.read_csv('./large_input/crs_2014_2023.csv')

    # Create text column
    df['text'] = df.progress_apply(create_unique_text, axis=1)

    # Load labeled texts from HF
    labels = load_dataset(f'alex-miller/crs-2014-2023-housing-labeled-phi4-reasoning{SUFFIX}', split='train')
    labels = labels.remove_columns(['PurposeCode'])
    labels = labels.to_pandas()

    df = df.merge(labels, how="left", on="text")
    df.to_csv(f"large_output/crs_2014_2023_phi4_reasoning_labeled{SUFFIX}.csv")



if __name__ == '__main__':
    main()
