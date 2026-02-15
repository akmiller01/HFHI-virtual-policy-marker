import os
import pandas as pd
from tqdm import tqdm
from hfhi_definitions import SUFFIX
# from wb_definitions import SUFFIX


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


def main(preprocessed_path=None, labeled_path=None, output_path=None, original_path=None):
    tqdm.pandas()

    if preprocessed_path is None:
        preprocessed_path = os.path.join('large_input', 'crs_2024_update_preprocessed.csv')

    # default original full CRS (from 1.0)
    if original_path is None:
        original_path = os.path.join('large_input', 'crs_2024_update.csv')

    base = os.path.splitext(os.path.basename(preprocessed_path))[0]
    if labeled_path is None:
        labeled_path = os.path.join('large_input', f"{base}_labeled{SUFFIX}.csv")
    if output_path is None:
        output_path = os.path.join('large_output', f"{base}_merged{SUFFIX}.csv")

    # Read original full CRS data (all columns) as strings to preserve original columns
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original CRS file not found: {original_path}")
    orig = pd.read_csv(original_path, dtype=str)

    # Ensure text column exists on original (recreate using same logic)
    if 'text' not in orig.columns or orig['text'].isnull().all():
        orig['text'] = orig.progress_apply(create_unique_text, axis=1)

    # Read labels produced by 3.0
    if not os.path.exists(labeled_path):
        raise FileNotFoundError(f"Labeled file not found: {labeled_path}")
    labels = pd.read_csv(labeled_path, dtype=str)

    # Normalize labels: drop duplicate text keys keeping first
    if 'text' in labels.columns:
        labels_by_text = labels.drop_duplicates(subset=['text']).copy()
    else:
        labels_by_text = labels.copy()

    # Merge labels onto original full CRS by text to recover labeled original rows
    orig_labeled = orig.merge(labels_by_text, how='left', on='text', suffixes=('', '_label'))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Write original merged output only
    orig_out_path = os.path.join('large_output', f"{base}_original_labeled{SUFFIX}.csv")
    orig_labeled.to_csv(orig_out_path, index=False)
    print(f'Wrote original merged output to: {orig_out_path}')


if __name__ == '__main__':
    main()
