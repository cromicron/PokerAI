import numpy as np
from tqdm import tqdm
import os
from scipy.stats import mode
import pickle

def compute_majority_mask_vectorized(data):
    """
    Compute majority masks for multiple rows of 7-element arrays.

    Args:
        data: A 2D NumPy array where each row has 7 elements.

    Returns:
        majority_values: A 1D array of majority values for each row, or -1 if no majority exists.
        masks: A 1D array of uint8 masks for each row, or 0x00 if no majority exists.
    """
    assert data.shape[1] == 7, "Each row must have exactly 7 elements"

    # Step 1: Count occurrences of each value per row
    unique_values = np.arange(256, dtype=np.uint8)  # Possible values (0-255 for uint8)
    counts = np.apply_along_axis(lambda row: np.bincount(row, minlength=256), axis=1, arr=data)

    # Step 2: Identify the majority value (if it occurs >= 5 times)
    majority_value_indices = np.argmax(counts, axis=1)  # Index of max count per row
    majority_counts = counts[np.arange(data.shape[0]), majority_value_indices]  # Max counts per row
    has_majority = majority_counts >= 5  # Boolean mask for rows with a majority value

    # Step 3: Assign majority values (-1 for rows with no majority)
    majority_values = np.where(has_majority, majority_value_indices, -1)

    # Step 4: Compute binary masks for rows with a majority value
    majority_binary_masks = (data == majority_value_indices[:, None]).astype(np.uint8)
    packed_masks = np.packbits(majority_binary_masks, axis=1, bitorder='big')[:, -1]  # Pack 7 bits into uint8

    # Step 5: Set packed mask to 0x00 for rows without a majority
    packed_masks[~has_majority] = 0x00

    return majority_values, packed_masks

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

if __name__ == "__main__":
    lookup_flush = {}
    path = "strength_evals"
    for filename in tqdm(os.listdir(path)):
        path_batch = os.path.join(path, filename)
        data = np.load(path_batch)
        cards = decode_cards(data[:, :7])
        suits = cards[:, :, 1].astype(int)
        flush = np.isin(data[:, -2].astype(int), (5, 8))
        flush_suits = mode(suits[flush], axis=1, keepdims=False).mode
        binary_masks = (suits[flush] == flush_suits[:, None]).astype(np.uint8)

        # Pack the binary masks into a compact uint8 representation
        packed_masks = np.packbits(binary_masks, axis=1, bitorder='big')[:, 0]
        suits_encoded = np.zeros(suits.shape[0], dtype=np.uint8)
        suits_encoded[flush] = packed_masks
        lookup_flush.update({tuple(suit_vals): encoded_suit for suit_vals, encoded_suit in zip(suits, suits_encoded)})

    with open("flush_lookup.pkl", "wb") as f:
        pickle.dump(lookup_flush, f)