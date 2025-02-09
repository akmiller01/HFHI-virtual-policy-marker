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
        "Housing": [
            "housing provision", "provision of shelter", "emergency shelter", "shelter provision",
            "upgrading housing", "improving housing", "housing improvements", "slum upgrading",
            "informal settlements", "basic services", "access to safe water", "sanitation services",
            "energy access", "waste management", "urban housing", "urban development", "city planning",
            "housing policy", "housing finance", "construction of housing", "home construction",
            "residential construction", "technical assistance for housing", "housing assistance",
            "housing support", "land tenure", "urban planning"
        ],
        "Homelessness": [
            "homeless shelter", "shelters for the homeless", "emergency shelters", "tent encampments",
            "encampments for the homeless", "unhoused population", "unsheltered individuals",
            "street homelessness", "rough sleeping", "temporary shelters", "transitional housing for homeless",
            "supportive housing for homeless"
        ],
        "Transitional": [
            "emergency housing", "crisis housing", "disaster housing", "temporary shelter",
            "temporary housing", "refugee shelter", "refugee housing", "refugee camp",
            "displacement camps", "internally displaced persons (IDP) camps", "post-disaster housing",
            "humanitarian shelter", "supportive transitional housing"
        ],
        "Incremental": [
            "housing sites", "serviced housing plots", "site and service schemes", "housing infrastructure",
            "housing assistance programs", "home repairs", "housing rehabilitation", "slum upgrading",
            "informal housing improvements", "upgrading informal settlements", "neighborhood integration",
            "community-driven housing improvements", "in-situ upgrading"
        ],
        "Social": [
            "public housing", "subsidized housing", "affordable housing", "low-cost housing",
            "nonprofit housing", "cooperative housing", "community housing", "community land trusts",
            "government-subsidized housing", "social housing programs", "mixed-income housing"
        ],
        "Market": [
            "home rentals", "rental housing", "mortgage financing", "mortgage assistance",
            "rent-to-own housing", "rent-to-own programs", "market-rate housing", "private housing market",
            "real estate development", "private sector housing", "homeownership programs"
        ]
    }
    main(query_topics)


