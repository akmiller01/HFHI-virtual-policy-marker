import os
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import pickle
from datasets import Dataset, load_dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

global DEVICE
global MODEL
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", device=DEVICE, trust_remote_code=True)


def main(queries, threshold=0.45):
    pickle_path = "large_input/gte-large-en-v1.5.pkl"
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    dataset = dataset.add_column('id', range(dataset.num_rows))
    pos = dataset.filter(lambda example: example['sector_code'] in [16030, 16040]).shuffle(seed=1337).select(range(50))
    neg = dataset.filter(lambda example: example['sector_code'] not in [16030, 16040]).shuffle(seed=1337).select(range(50))
    dataset = concatenate_datasets([neg, pos])
    sentences = dataset["text"]


    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as pickle_file:
            file_embeddings = pickle.load(pickle_file)
    else:
        file_embeddings = list()
        for sentence in tqdm(sentences):
            embedding = MODEL.encode(sentence)
            file_embeddings.append(embedding)
        with open(pickle_path, 'wb') as pickle_file:
            pickle.dump(file_embeddings, pickle_file)

    query_embeddings = MODEL.encode(queries)
    ranks = np.zeros(len(file_embeddings))
    query_maxes = list()
    for i, embedding in enumerate(file_embeddings):
        similarity = cos_sim(query_embeddings, embedding)
        max_sim_index = np.argmax(similarity).tolist()
        ranks[i] = similarity.mean()
        query_maxes.append(queries[max_sim_index])

    rank_dataset = Dataset.from_dict(
        {
            'id': dataset['id'],
            'text': sentences,
            'sector_code': dataset['sector_code'],
            'rank': ranks,
            'max_query': query_maxes
        }
    )
    matching_dataset = rank_dataset.filter(lambda example: example['rank'] >= threshold)
    rank_dataset.to_csv("large_input/embedding_filtered_crs_2014_2023.csv")


if __name__ == '__main__':
    query_str = "housing, housing policy, housing finance, habitability, tents for the homeless, encampments for the homeless, homeless shelters, emergency shelters, refugee shelters, refugee camps, temporary supportive housing, housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, neighborhood integration, community land trusts, cooperative housing, public housing, subsidized home-rental, subsidized mortgages, rent-to-own housing, market-rate housing"
    queries = query_str.split(", ")
    main(queries)