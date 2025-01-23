import os
from glob import glob
import pandas as pd
import json
from tqdm import tqdm
import click


global OUT_FOLDER
# OUT_FOLDER = 'large_input/gpt_batch_files/crs_2014_2023'
OUT_FOLDER = 'large_input/gpt_batch_files/crs_2014_2023_addvague'


def correct_columns(key):
    corrections = {
        'housing_generic': 'housing_general',
        'homeliness_support': 'homelessness_support',
        # 'market rent_own': 'market_rent_own',
        'market rent_own': 'market_rent_own_housing',
        'rural:false,': 'rural',
        'rural:true,': 'rural',
        'climate_adaption': 'climate_adaptation',
        'clima..._adaptation': 'climate_adaptation',
        'clima_mitigation': 'climate_mitigation',
    }
    key = key.strip().lower()
    if key in corrections.keys():
        return corrections[key]
    return key


def correct_values(value):
    corrections = {
        'true': True,
        'false': False,
        '': False
    }
    if type(value) == list:
        value = value[0]
    if value in corrections.keys():
        return corrections[value]
    if pd.isna(value):
        return False
    return value


def manual_entry():
    results = dict()
    keys = [
        "vague_or_short",
        "housing_general",
        "homelessness_support",
        "transitional_housing",
        "incremental_housing",
        "social_housing",
        # "market_rent_own",
        'market_rent_own_housing',
        "urban",
        "rural",
        "climate_adaptation",
        "climate_mitigation"
    ]
    for key in keys:
        results[key] = click.confirm(key, default=False)
    return results


def main():
    manual_count = 0
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
                results_message = response_json['response']['body']['choices'][0]['message']
                if 'function_call' in results_message.keys():
                    results_str = results_message['function_call']['arguments']
                else:
                    results_str = results_message['content']
                try:
                    results = json.loads(results_str)
                    results = {correct_columns(key): correct_values(value) for key, value in results.items()}
                except json.decoder.JSONDecodeError:
                    print(batch_data[batch_data['id'] == id]['text'].values.tolist()[0])
                    print(results_str)
                    results = manual_entry()
                    manual_count += 1
                results['id'] = id
                result_records.append(results)
        result_df = pd.DataFrame.from_records(result_records)
        result_df["climate_adaptation"] = result_df["climate_adaptation"].astype(bool).fillna(False)
        result_df = result_df.astype(int)
        batch_data = batch_data.merge(result_df, on="id")
        data_frames.append(batch_data)

    all_data = pd.concat(data_frames)
    all_data.to_csv('large_input/crs_2014_2023_gpt_batched_vague.csv', index=False)
    print(f"Manually answered: {manual_count}")

    previous_data = pd.read_csv('large_input/crs_2014_2023_gpt_batched.csv')
    previous_data = previous_data.drop(
        ['housing_general', 'homelessness_support', 'transitional_housing', 'incremental_housing', 'social_housing', 'market_rent_own'],
        axis=1
    )
    new_data = all_data[['id', 'vague_or_short', 'housing_general', 'homelessness_support', 'transitional_housing', 'incremental_housing', 'social_housing', 'market_rent_own_housing']]
    merged_data = pd.merge(previous_data, new_data, how="left", on="id")
    merged_data = merged_data.fillna(0)
    merged_data.to_csv('large_input/crs_2014_2023_gpt_batched2.csv', index=False)


if __name__ == '__main__':
    main()
