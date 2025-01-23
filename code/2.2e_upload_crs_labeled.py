from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
import os


def main():
    # Load
    dataset = load_dataset("csv", data_files="large_input/crs_2014_2023_gpt_batched2.csv", split="train")
    # Push to HF
    dataset.push_to_hub('alex-miller/crs-2014-2023-housing-labeled')


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()
