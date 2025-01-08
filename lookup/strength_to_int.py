import numpy as np
import os
from tqdm import tqdm

all_strengths = set([])

def decode_cards(encoded_batch):
    """
    Decode a batch of encoded cards back into [rank, suit].
    - encoded_batch: NumPy array of shape (n, 7) with encoded integers.
    Returns:
    - Decoded batch: NumPy array of shape (n, 7, 2) with [rank, suit].
    """
    ranks = (encoded_batch // 4) + 2
    suits = encoded_batch % 4
    return np.stack([ranks, suits], axis=-1)


path = "strength_evals"
for filename in tqdm(os.listdir(path)):
    path_batch = os.path.join(path, filename)
    data = np.load(path_batch)
    strengths = np.round(data[:, 7], 10)
    all_strengths.update(set(np.unique(strengths)))
strength_rank_dict = {strength: rank for rank, strength in enumerate(sorted(all_strengths))}

for filename in tqdm(os.listdir(path)):
    path_batch = os.path.join(path, filename)
    data = np.load(path_batch)
    strengths = np.round(data[:, 7], 10)
    ranks = np.array([strength_rank_dict[strength]for strength in strengths]).reshape(-1, 1)
    data = np.hstack([data, ranks])
    np.save(path_batch, data)