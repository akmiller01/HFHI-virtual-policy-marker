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

    dataset.push_to_hub('alex-miller/crs-2014-2023-housing-similarity')


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    query_topics = {
        "Housing general": [
            "housing provision", "shelter provision", "housing construction", "shelter construction",
            "eco-friendly building", "green building", "sustainable housing", "housing policy",
            "technical assistance for housing", "housing finance", "affordable home loans",
            "residential construction", "housing development", "urban housing", "rural housing",
            "resilient housing", "inclusive housing", "housing access", "dwelling construction"
        ],
        "Homelessness": [
            "homeless shelters", "emergency shelters", "homeless encampments", "tent cities",
            "tents for the homeless", "transitional housing for homeless", "unsheltered population",
            "temporary housing for homeless", "street homelessness", "permanent supportive housing",
            "housing first programs", "shelter programs for homeless"
        ],
        "Transitional housing": [
            "emergency housing", "crisis housing", "disaster shelter", "disaster housing",
            "temporary housing", "refugee shelters", "humanitarian shelters",
            "temporary supportive housing", "emergency accommodation", "post-disaster housing",
            "shelters for displaced persons", "temporary accommodation for refugees"
        ],
        "Incremental housing": [
            "housing improvement", "home repairs", "slum upgrading", "basic services for housing",
            "housing technical assistance", "incremental housing", "informal settlement upgrading",
            "water services for housing", "sanitation for housing", "energy services for housing",
            "upgrading informal housing"
        ],
        "Social housing": [
            "Community Land Trust", "CLT", "cooperative housing", "public housing",
            "affordable housing", "low-cost housing", "social housing", "nonprofit housing",
            "subsidized housing", "government-funded housing", "permanently affordable housing",
            "housing cooperatives", "shared-equity housing", "municipal housing",
            "state-funded housing"
        ],
        "Market housing": [
            "home rental", "rental housing", "mortgages", "home financing", "rent-to-own housing",
            "market-rate housing", "private housing market", "real estate market",
            "homeownership programs", "property development", "private sector housing",
            "residential real estate", "housing investment", "landlord-tenant rental"
        ]
    }
    main(query_topics)


