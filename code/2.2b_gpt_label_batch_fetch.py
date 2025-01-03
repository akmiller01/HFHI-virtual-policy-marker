import os
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


def main():
    batch_id_files = glob(os.path.join(OUT_FOLDER, '*.txt'))
    for batch_id_file in batch_id_files:
        batch_id_filename, _ = os.path.splitext(batch_id_file)
        result_filename = f'{batch_id_filename}_completed.jsonl'
        if not os.path.exists(result_filename):
            with open(batch_id_file, 'r') as id_file:
                batch_id = id_file.read()
                batch = CLIENT.batches.retrieve(batch_id)
                if batch.status=='completed':
                    output_file_id = batch.output_file_id
                    file_response = CLIENT.files.content(output_file_id)
                    with open(result_filename, 'w') as result_file:
                        result_file.write(file_response.text)
                else:
                    percentage_complete = batch.request_counts.completed / batch.request_counts.total
                    percentage_complete_str = round(percentage_complete * 100)
                    print(f'{os.path.basename(batch_id_filename)} is {percentage_complete_str}% completed.')


if __name__ == '__main__':
    main()
