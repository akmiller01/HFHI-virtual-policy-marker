# ! pip install tqdm torch sentence-transformers numpy datasets python-dotenv huggingface_hub tf-keras

import os
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login


global DEVICE
global MODEL
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", device=DEVICE, trust_remote_code=True)


def main(queries):
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    text_embeddings = MODEL.encode(dataset["text"], batch_size=512, show_progress_bar=True, normalize_embeddings=True)
    query_embeddings = MODEL.encode(queries, normalize_embeddings=True)
    similarities = MODEL.similarity(query_embeddings, text_embeddings)

    for i in range(0, len(queries)):
        query = queries[i]
        col_name = "{}_similarity".format(query.replace(' ', '_').replace('-', '_'))
        dataset = dataset.add_column(col_name, similarities[i,:].tolist())

    dataset.push_to_hub('alex-miller/crs-2014-2023-housing-similarity')


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    query_str = "housing, housing policy, housing finance, habitability, homelessness, tents, encampments, shelters, emergency shelters, refugee shelters, temporary supportive housing, housing sites, housing services, housing technical assistance, slums, slum upgrading, housing structural repairs, neighborhood integration, community land trusts, cooperative housing, public housing, home-rental, homeownership, mortgages, rent-to-own housing, market-rate housing"
    queries = query_str.split(", ")
    main(queries)


