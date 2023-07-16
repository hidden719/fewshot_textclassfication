import collections

import numpy as np
import random
from typing import List
import torch


def random_sample_cls(sentences: List[str], labels: List[str], n_support: int, n_query: int, label: str):
    """
    Randomly samples Ns examples as support set and Nq as Query set
    """
    data = [sentences[i] for i, lab in enumerate(labels) if lab == label]
    perm = torch.randperm(len(data))
    idx = perm[:n_support]
    support = [data[i] for i in idx]
    idx = perm[n_support: n_support + n_query]
    query = [data[i] for i in idx]

    return support, query


def create_episode(data_dict, n_support, n_classes, n_query, n_unlabeled=0, n_augment=0):
    n_classes = min(n_classes, len(data_dict.keys()))
    rand_keys = np.random.choice(list(data_dict.keys()), n_classes, replace=False)

    assert min([len(val) for val in data_dict.values()]) >= n_support + n_query + n_unlabeled

    for key, val in data_dict.items():
        random.shuffle(val)

    episode = {
        "xs": [
            [data_dict[k][i] for i in range(n_support)] for k in rand_keys
        ],
        "xq": [
            [data_dict[k][n_support + i] for i in range(n_query)] for k in rand_keys
        ]
    }

    if n_unlabeled:
        episode['xu'] = [
            item for k in rand_keys for item in data_dict[k][n_support + n_query:n_support + n_query + 10]
        ]

    if n_augment:
        augmentations = list()
        already_done = list()
        for i in range(n_augment):
            # Draw a random label
            key = random.choice(list(data_dict.keys()))
            # Draw a random data index
            ix = random.choice(range(len(data_dict[key])))
            # If already used, re-sample
            while (key, ix) in already_done:
                key = random.choice(list(data_dict.keys()))
                ix = random.choice(range(len(data_dict[key])))
            already_done.append((key, ix))
            if "augmentations" not in data_dict[key][ix]:
                raise KeyError(f"Input data {data_dict[key][ix]} does not contain any augmentations / is not properly formatted.")
            augmentations.append((
                data_dict[key][ix]["sentence"],
                [item["text"] for item in data_dict[key][ix]["augmentations"]]
            ))
        episode["x_augment"] = augmentations

    return episode
