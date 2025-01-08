from tqdm import tqdm
import numpy as np
import pickle
from create_lookup_suits import decode_cards
import os

lookup = {}

with open("flush_lookup.pkl", "rb") as f:
    suit_dict = pickle.load(f)


def encode_hands_array(ranks, suit_info):
    # Adjust ranks to range 0-12
    adjusted_ranks = ranks - 2

    # Pack ranks into a single integer
    rank_encoded = (
        (adjusted_ranks[:, 0] << 24) | (adjusted_ranks[:, 1] << 20) |
        (adjusted_ranks[:, 2] << 16) | (adjusted_ranks[:, 3] << 12) |
        (adjusted_ranks[:, 4] << 8) | (adjusted_ranks[:, 5] << 4) |
        adjusted_ranks[:, 6]
    )  # Shape: (n,)

    # Combine rank encoding and suit info
    combined = (rank_encoded << 10) + suit_info
    return combined

def encode_hand(ranks, suit_info):
    adjusted_ranks = [rank - 2 for rank in ranks]

    # Pack ranks into a single integer
    rank_encoded = (
        (adjusted_ranks[0] << 24) | (adjusted_ranks[1] << 20) |
        (adjusted_ranks[2] << 16) | (adjusted_ranks[3] << 12) |
        (adjusted_ranks[4] << 8) | (adjusted_ranks[5] << 4) |
        adjusted_ranks[6]
    )

    # Combine rank encoding and suit info
    combined = (rank_encoded << 10) + suit_info
    return combined

if __name__ == "__main__":
    path = "strength_evals"
    with tqdm(os.listdir(path), desc="Processing files", unit="file") as pbar:
        lookup = {}
        for filename in pbar:
            path_batch = os.path.join(path, filename)
            data = np.load(path_batch)
            cards = decode_cards(data[:, :7])

            suit_info = np.array([suit_dict[tuple(card[:, 1])] for card in cards])

            # Encode the hands
            ranks = cards[:, :, 0].astype(np.uint64)
            encoded_hands = encode_hands_array(ranks, suit_info.astype(np.uint64))
            values = data[:, -1].astype(np.uint16)

            # Step 1: Get unique encodings and their indices
            unique_encodings, inverse_indices = np.unique(encoded_hands, return_inverse=True)

            # Step 2: Group values by unique encodings
            consistent = True
            for encoding_index in range(unique_encodings.size):
                # Extract indices corresponding to the current encoding
                indices = np.where(inverse_indices == encoding_index)[0]
                # Check if all values for this encoding are the same
                if not np.all(values[indices] == values[indices][0]):
                    consistent = False
                    print("Error: Inconsistent values for encoding")
                    break
                else:
                    lookup[unique_encodings[encoding_index]] = values[indices][0]

            # Update tqdm description with the current size of the lookup dict
            pbar.set_description(f"Processing files (Lookup size: {len(lookup)})")

    with open("lookup.pkl", "wb") as f:
        pickle.dump(lookup, f)

