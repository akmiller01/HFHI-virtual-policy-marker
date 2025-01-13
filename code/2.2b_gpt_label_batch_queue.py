import os
import time
from glob import glob
from openai import OpenAI
from dotenv import load_dotenv

global OUT_FOLDER
OUT_FOLDER = 'large_input/gpt_batch_files/crs_2014_2023'

load_dotenv()
global CLIENT
CLIENT = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

global MODEL
MODEL = "gpt-4o-mini"

global SLEEP
SLEEP = 1800


def main():
    incomplete = True

    while incomplete:
        # If a text file exists, a process is running.
        # Check to see if it's done
        ## If it's done, download it and delete txt file
        ## If it's not done, sleep
        txt_files = glob(os.path.join(OUT_FOLDER, '*.txt'))
        if len(txt_files) > 1:
            print("More than one txt file detected. Breaking loop.")
            break
        if len(txt_files) == 1:
            batch_id_file = txt_files[0]
            batch_id_filename, _ = os.path.splitext(batch_id_file)
            result_filename = f'{batch_id_filename}_completed.jsonl'
            with open(batch_id_file, 'r') as id_file:
                batch_id = id_file.read()
                batch = CLIENT.batches.retrieve(batch_id)
                if batch.status == 'completed':
                    output_file_id = batch.output_file_id
                    file_response = CLIENT.files.content(output_file_id)
                    with open(result_filename, 'w') as result_file:
                        result_file.write(file_response.text)
                    os.remove(batch_id_file)
                    print(f'{os.path.basename(batch_id_filename)} is complete! Uploading next file if necessary...')
                    continue
                elif batch.status in ['validating', 'in_progress', 'finalizing']:
                    if batch.request_counts.total > 0:
                        percentage_complete = batch.request_counts.completed / batch.request_counts.total
                        percentage_complete_str = round(percentage_complete * 100)
                        print(f'{os.path.basename(batch_id_filename)} is {percentage_complete_str}% completed.')
                    else:
                        print(f'{os.path.basename(batch_id_filename)} is 0% completed.')
                    print(f"SLEEPING {SLEEP}")
                    time.sleep(SLEEP)
                    continue
                else:
                    print(f'{os.path.basename(batch_id_filename)} status is {batch.status}.')
                    break

        # No txt files below this point
        # Iterate through batch files, looking for those without a result file
        batch_files = glob(os.path.join(OUT_FOLDER, '*.jsonl'))
        batch_files = [batch_file for batch_file in batch_files if not batch_file.endswith('_completed.jsonl')]
        for batch_file in batch_files:
            batch_filename, _ = os.path.splitext(batch_file)
            batch_basename = os.path.basename(batch_filename)
            result_filename = f'{batch_filename}_completed.jsonl'
            if not os.path.exists(result_filename):
                print(f'{batch_basename} does not yet have a result file. Uploading...')
                # Upload file
                batch_file_response = CLIENT.files.create(
                    file=open(batch_file, "rb"),
                    purpose="batch"
                )
                batch_file_response_id = batch_file_response.id
                # Create batch process
                batch_process_response = CLIENT.batches.create(
                    input_file_id=batch_file_response_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": f"CRS labeling {batch_basename}"
                    }
                )
                # Save ID in txt file
                batch_process_response_id = batch_process_response.id
                batch_id_filename = f'{batch_filename}.txt'
                with open(batch_id_filename, 'w') as batch_id_file:
                    batch_id_file.write(batch_process_response_id)
                print(f"SLEEPING {SLEEP}")
                time.sleep(SLEEP)
                break # Break out of loop after the first upload

        # Process is complete when every batch-0.jsonl has a corresponding batch-0_completed.jsonl
        batch_files = glob(os.path.join(OUT_FOLDER, '*.jsonl'))
        batch_files = [batch_file for batch_file in batch_files if not batch_file.endswith('_completed.jsonl')]
        batch_completed_files = glob(os.path.join(OUT_FOLDER, '*_completed.jsonl'))
        print(f'Entire process is {round((len(batch_completed_files)/len(batch_files))*100, 1)}% complete.')
        incomplete = len(batch_files) > len(batch_completed_files)


if __name__ == '__main__':
    main()
