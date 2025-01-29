import os
import numpy as np
import click
import scipy.stats as stats
import math
from datasets import load_dataset
import pickle

def wilson_interval_with_fpc(accuracy, sample_size, population_size, confidence_level=0.95):
    """
    Calculate the Wilson confidence interval with finite population correction.

    Parameters:
    - accuracy (float): Accuracy percentage (e.g., 0.85 for 85%).
    - sample_size (int): Number of samples.
    - population_size (int): Total population size.
    - confidence_level (float): Confidence level (default is 0.95).

    Returns:
    - tuple: (lower_bound, upper_bound) of the confidence interval.
    """
    # Convert accuracy percentage to proportion
    p = accuracy

    # Compute the z-value for the confidence level
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Calculate the finite population correction factor
    fpc = math.sqrt((population_size - sample_size) / (population_size - 1)) if population_size > sample_size else 1

    # Wilson interval calculation
    n = sample_size
    denominator = 1 + (z**2 / n)
    center = (p + (z**2 / (2 * n))) / denominator
    margin = (z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denominator

    # Apply finite population correction
    margin *= fpc

    lower_bound = max(0, center - margin).tolist()
    upper_bound = min(1, center + margin).tolist()

    return lower_bound, upper_bound


def main():
    cache = dict()
    pickle_path = "large_input/accuracy_cache.pkl"
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as handle:
            cache = pickle.load(handle)
    else:
        with open(pickle_path, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sample_size = 5
    dataset = load_dataset("alex-miller/crs-2014-2023-housing-labeled", split="train")
    dimension_dict = {
        "housing": "directly or indirectly relates to any of the following: housing, housing policy, housing finance, habitability, tents for the homeless, encampments for the homeless, homeless shelters, emergency shelters, refugee shelters, refugee camps, temporary supportive housing, housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, neighborhood integration, community land trusts, cooperative housing, public housing, subsidized home-rental, subsidized mortgages, rent-to-own housing, or market-rate housing",
        "homelessness": "explicitly describes tents for the homeless, encampments for the homeless, or homeless shelters",
        "transitional": "explicitly describes emergency shelters, refugee shelters, refugee camps, or temporary supportive housing",
        "incremental": "explicitly describes housing sites, housing services, housing technical assistance, slum upgrading, housing structural repairs, or neighborhood integration",
        "social": "explicitly describes community land trusts, cooperative housing, or public housing",
        "market": "explicitly describes subsidized home-rental, subsidized mortgages, rent-to-own housing, or market-rate housing",
        "urban_rural": "explicitly describes activities in specific urban locations, specific rural locations, both urban and rural locations, or neither"
    }

    for key in dimension_dict.keys():
        thoughts_key = f'{key}_thoughts'
        answer_key = f'{key}_answer'
        if key not in cache.keys():
            cache[key] = dict()
        print(f"\n\nColumn: {key}")
        definition = dimension_dict[key]
        print(f"Definition: {definition}")

        if key == "urban_rural":
            pos = dataset.filter(lambda example: example[answer_key] in ['Urban', 'Rural', 'Both'])
            neg = dataset.filter(lambda example: example[answer_key] == 'Neither')
        else:
            pos = dataset.filter(lambda example: example[answer_key] == True)
            neg = dataset.filter(lambda example: example[answer_key] == False)

        pos_pop = pos.num_rows
        neg_pop = neg.num_rows
        all_pop = dataset.num_rows

        pos_sample = pos.shuffle(seed=42).select(range(sample_size))
        neg_sample = neg.shuffle(seed=42).select(range(sample_size))

        pos_corrects = list()
        pos_sample_size = sample_size
        for i, text in enumerate(pos_sample['text']):
            if key in cache.keys() and text in cache[key].keys():
                manual_entry = cache[key][text]
            else:
                thoughts = pos_sample[thoughts_key][i]
                print(f"{i + 1}. {text}\nModel thoughts: {thoughts}")
                try:
                    manual_entry = int(click.confirm(key, default=True))
                    cache[key][text] = manual_entry
                    with open(pickle_path, 'wb') as handle:
                        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except click.exceptions.Abort:
                    pos_sample_size -= 1
                    continue
            pos_corrects.append(int(manual_entry == True))


        pos_sample_accuracy = np.mean(pos_corrects).tolist()
        pos_sample_accuracy_round = round(pos_sample_accuracy * 100, 1)
        pos_lower, pos_upper = wilson_interval_with_fpc(pos_sample_accuracy, pos_sample_size, pos_pop)
        pos_lower_round = round(pos_lower * 100, 1)
        pos_upper_round = round(pos_upper * 100, 1)

        neg_corrects = list()
        neg_sample_size = sample_size
        for i, text in enumerate(neg_sample['text']):
            if key in cache.keys() and text in cache[key].keys():
                manual_entry = cache[key][text]
            else:
                thoughts = neg_sample[thoughts_key][i]
                print(f"{i + 1}. {text}\nModel thoughts: {thoughts}")
                try:
                    manual_entry = int(click.confirm(key, default=False))
                    cache[key][text] = manual_entry
                    with open(pickle_path, 'wb') as handle:
                        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except click.exceptions.Abort:
                    neg_sample_size -= 1
                    continue
            automated_entry = neg_sample[key][i]
            neg_corrects.append(int(manual_entry == False))

        neg_sample_accuracy = np.mean(neg_corrects).tolist()
        neg_sample_accuracy_round = round(neg_sample_accuracy * 100, 1)
        neg_lower, neg_upper = wilson_interval_with_fpc(neg_sample_accuracy, neg_sample_size, neg_pop)
        neg_lower_round = round(neg_lower * 100, 1)
        neg_upper_round = round(neg_upper * 100, 1)

        all_corrects = pos_corrects + neg_corrects
        all_sample_size = pos_sample_size + neg_sample_size
        all_sample_accuracy = np.mean(all_corrects).tolist()
        all_sample_accuracy_round = round(all_sample_accuracy * 100, 1)
        all_lower, all_upper = wilson_interval_with_fpc(all_sample_accuracy, all_sample_size, all_pop)
        all_lower_round = round(all_lower * 100, 1)
        all_upper_round = round(all_upper * 100, 1)
        print(f"Accuracy: {all_sample_accuracy_round}%; {all_lower_round} - {all_upper_round}")
        print(f"Recall: {pos_sample_accuracy_round}%; {pos_lower_round} - {pos_upper_round}")
        print(f"Precision: {neg_sample_accuracy_round}%; {neg_lower_round} - {neg_upper_round}")


if __name__ == '__main__':
    main()
