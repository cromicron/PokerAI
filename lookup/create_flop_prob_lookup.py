import numpy as np
from copy import deepcopy
import pickle
import numpy as np

def encode_hand(hand):
    """
    Encodes a list of 5 two-tuples into a single integer.

    :param tuples: List of 5 tuples where each tuple contains:
                   - First element: int between 2 and 14 (inclusive)
                   - Second element: int between 0 and 3 (inclusive)
    :return: Encoded integer
    """
    encoded = 0
    for i, (first, second) in enumerate(hand):
        # Shift the encoded integer to make space for the next tuple
        encoded <<= 6  # Each tuple takes 6 bits
        # Pack the first and second elements into the integer
        encoded |= ((first - 2) << 2) | second  # Use 4 bits for first, 2 bits for second
    return encoded

def encode_hand_array(hands):
    """
    Encodes an n*5*2 NumPy array into a 1D NumPy array of integers.

    :param array: NumPy array of shape (n, 5, 2),
                  where:
                  - First value of each tuple is in range [2, 14]
                  - Second value of each tuple is in range [0, 3]
    :return: Encoded NumPy array of shape (n,)
    """
    n = hands.shape[0]
    encoded_array = np.zeros(n, dtype=np.uint32)  # Use uint64 for large integers
    for i in range(5):  # Loop over the 5 tuples in each row
        first = hands[:, i, 0] - 2  # Normalize first value to range [0, 12]
        second = hands[:, i, 1]     # Second value is already in range [0, 3]
        encoded_array = (encoded_array << 6) | ((first << 2) | second)  # Pack 6 bits per tuple
    return encoded_array.astype(np.uint32)

file_probs = "../training_scripts/flop_probs.npy"
file_hands = "../training_scripts/hero_flop.npy"

hands = np.load(file_hands)
probs = np.load(file_probs)

# Assuming hands is your 3D array
preflop = hands[:, :2, :]
flop = hands[:, 2:, :]

# Sort preflop by both the first and second elements
n = preflop.shape[0]
sort_indices_hole = np.lexsort((preflop[:, :, 1], preflop[:, :, 0]), axis=1)
preflop = preflop[np.arange(n)[:, None], sort_indices_hole]

# Sort flop by both the first and second elements
sort_indices_flop = np.lexsort((flop[:, :, 1], flop[:, :, 0]), axis=1)
flop = flop[np.arange(n)[:, None], sort_indices_flop]

# Concatenate preflop and flop back together
hands = np.concatenate([preflop, flop], axis=1)


# find first card that is not 0 or 1 and encode it with 4, encode other with 5
# then drop duplicates
suits = hands[:, :, 1]
suits_recoded = np.zeros_like(suits)
suits_recoded[:,1] = np.where(suits[:,1]==suits[:,0], 0, 1)
suits_recoded[:,2] = np.where(suits[:,2]== suits[:,0], 0, np.where(
    suits[:,2]==suits[:,1], suits_recoded[:,1], suits_recoded.max(axis=1)+1
))
suits_recoded[:,3] = np.where(suits[:,3]== suits[:,0], 0, np.where(
    suits[:,3]==suits[:,1], suits_recoded[:,1], np.where(
        suits[:,3]==suits[:,2],suits_recoded[:,2],suits_recoded.max(axis=1)+1)
))
suits_recoded[:,4] = np.where(suits[:,4]== suits[:,0], 0, np.where(
    suits[:,4]==suits[:,1], suits_recoded[:,1], np.where(
        suits[:,4]==suits[:,2],suits_recoded[:,2], np.where(
            suits[:, 4]==suits[:,3], suits_recoded[:,3], suits_recoded.max(axis=1)+1
        ))
))
hands[:,:,1] = suits_recoded
unique_hands, i = np.unique(hands, axis=0, return_index=True)
unique_probs = probs[i]


hands_encoded = encode_hand_array(unique_hands)

lookup = {card: tuple(probs) for card, probs in zip(hands_encoded, unique_probs.astype(np.float32))}

with open("lookup_probs_flop.pkl", "wb") as f:
    pickle.dump(lookup, f)