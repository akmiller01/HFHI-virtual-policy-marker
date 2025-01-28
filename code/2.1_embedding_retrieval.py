# ! pip install tqdm torch sentence-transformers numpy datasets python-dotenv huggingface_hub tf-keras

import os
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login


global DEVICE
global MODEL
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = SentenceTransformer("BAAI/bge-multilingual-gemma2", model_kwargs={"torch_dtype": torch.float16}, device=DEVICE)


def main(queries):
    # Prepare a prompt given an instruction
    instruction = 'Given a short topic, retrieve text that matches the topic.'
    prompt = f'<instruct>{instruction}\n<query>'
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    query_embeddings = MODEL.encode(queries, prompt=prompt)
    text_embeddings = MODEL.encode(dataset["text"], batch_size=256, show_progress_bar=True)
    similarities = MODEL.similarity(query_embeddings, text_embeddings)
    for i in range(0, len(queries)):
        query = queries[i]
        col_name = "{}_similarity".format(query.replace(' ', '_').replace('-', '_'))
        dataset = dataset.add_column(col_name, similarities[:,i])
    dataset.push_to_hub('alex-miller/crs-2014-2023-housing-similarity')


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    query_str = "housing, housing policy, housing finance, habitability, homelessness, tents, encampments, shelters, emergency shelters, refugee shelters, temporary supportive housing, housing sites, housing services, housing technical assistance, slums, slum upgrading, housing structural repairs, neighborhood integration, community land trusts, cooperative housing, public housing, home-rental, homeownership, mortgages, rent-to-own housing, market-rate housing"
    queries = query_str.split(", ")
    main(queries)