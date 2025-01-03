import os
from glob import glob
import pandas as pd
import json
from tqdm import tqdm

global OUT_FOLDER
OUT_FOLDER = 'large_input/gpt_batch_files/crs_2014_2023'


def main():
    data_frames = list()
    completed_files = glob(os.path.join(OUT_FOLDER, '*_completed.jsonl'))
    for completed_file in tqdm(completed_files):
        completed_filename, _ = os.path.splitext(os.path.basename(completed_file))
        batch_filename = completed_filename.split("_")[0]
        batch_data_filename  = os.path.join(OUT_FOLDER, f'{batch_filename}.csv')
        batch_data = pd.read_csv(batch_data_filename)
        result_records = list()
        with open(completed_file, 'r') as completed_file_reader:
            for line in completed_file_reader:
                response_json = json.loads(line)
                custom_id = response_json['custom_id']
                id = int(custom_id.split("-")[1])
                results_str = response_json['response']['body']['choices'][0]['message']['function_call']['arguments']
                results = json.loads(results_str)
                results['id'] = id
                result_records.append(results)
        result_df = pd.DataFrame.from_records(result_records)
        batch_data = batch_data.merge(result_df, on="id")
        data_frames.append(batch_data)

    all_data = pd.concat(data_frames)
    all_data.to_csv('large_input/crs_2014_2023_gpt_batched.csv', index=False)



if __name__ == '__main__':
    main()
