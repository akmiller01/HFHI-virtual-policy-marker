from transformers import pipeline
from datasets import load_dataset, concatenate_datasets


def main():
    # Load data
    dataset = load_dataset('alex-miller/crs-2014-2023', split='train')
    pos = dataset.filter(lambda row: row['sector_code'] in [16030, 16040])
    neg = dataset.filter(lambda row: row['sector_code'] not in [16030, 16040])
    neg = neg.shuffle(seed=1337).select(range(pos.num_rows))
    dataset = concatenate_datasets([pos, neg])

    # Prep model args
    classes_verbalized = [
        # Housing continuum
        'Homelessness support: Tents and encampments. Homeless shelters.',
        'Transitional housing: Emergency and refugee shelters and camps. Semi-permanent supportive housing.',
        'Incremental housing: Sites, services and technical assistance. Slum upgrading and structural repairs. Neighborhood integration.',
        'Social housing: Community Land Trusts. Cooperative housing. Public housing.',
        'Market rent & own: Social and subsidized rental. Supported homeownership (first-time, rent-to-own). Market-rate affordable housing.',
        # Urban/rural
        'Urban',
        'Rural',
        # Climate
        'Climate adaptation',
        'Climate mitigation'
    ]

    # Set up classifier
    zeroshot_classifier = pipeline("zero-shot-classification", model="tasksource/ModernBERT-base-nli", max_length=512)

    # Test classifier
    def inference_classifier(example):
        output = zeroshot_classifier(example['text'], classes_verbalized, multi_label=True)
        for i, label in enumerate(output['labels']):
            example[label] = output['scores'][i]
        return example

    # Inference
    dataset = dataset.map(inference_classifier)
    dataset.to_csv("large_input/crs_2014_2023_zs.csv")


if __name__ == '__main__':
    main()