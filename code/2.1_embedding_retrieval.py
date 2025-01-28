from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

global DEVICE
global MODEL
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", device=DEVICE, trust_remote_code=True)


def main(queries):
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    dataset = dataset.shuffle(seed=1337).select(range(10))
    sentences = dataset["text"]

    file_embeddings = list()
    for sentence in tqdm(sentences):
        embedding = MODEL.encode(sentence)
        file_embeddings.append(embedding)

    query_embeddings = MODEL.encode(queries)
    similarities = np.zeros(len(file_embeddings))
    query_maxes = list()
    for i, embedding in enumerate(file_embeddings):
        similarity = cos_sim(query_embeddings, embedding)
        max_sim_index = np.argmax(similarity).tolist()
        similarities[i] = similarity.mean()
        query_maxes.append(queries[max_sim_index])

    dataset = dataset.add_column("housing_similarity", similarities)
    dataset = dataset.add_column("max_query", query_maxes)

    dataset.to_csv("large_input/embedding_filtered_crs_2014_2023.csv")


if __name__ == '__main__':
    query_str = "housing, housing policy, housing finance, habitability, tents for the homeless, encampments for the homeless, homeless shelters, emergency shelters, refugee shelters, refugee camps, temporary supportive housing, housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, neighborhood integration, community land trusts, cooperative housing, public housing, subsidized home-rental, subsidized mortgages, rent-to-own housing, market-rate housing"
    queries = query_str.split(", ")
    main(queries)