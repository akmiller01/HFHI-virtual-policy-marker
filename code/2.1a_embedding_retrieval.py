# ! pip install tqdm torch sentence-transformers numpy datasets python-dotenv huggingface_hub tf-keras

import os
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from hfhi_definitions import SUFFIX, RETRIEVAL_TOPICS
from util_self_termination import main as self_terminate


global DEVICE
global MODEL
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", device=DEVICE, trust_remote_code=True)


def main(query_dictionary):
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    text_embeddings = MODEL.encode(dataset["text"], batch_size=512, show_progress_bar=True, normalize_embeddings=True)
    for topic in query_dictionary:
        queries = query_dictionary[topic]
        query_embeddings = MODEL.encode(queries, normalize_embeddings=True)
        similarities = MODEL.similarity(query_embeddings, text_embeddings)
        aggregated_similarities = 0.7 * similarities.max(axis=0).values + 0.3 * similarities.mean(axis=0)
        dataset = dataset.add_column(topic, aggregated_similarities.tolist())

    dataset.push_to_hub(f'alex-miller/crs-2014-2023-housing-similarity{SUFFIX}')


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main(RETRIEVAL_TOPICS)
    self_terminate()


