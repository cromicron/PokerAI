import numpy as np
from PokerGame.NLHoldem import Deck
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for the progress bar
import math
from lookup.HandComparatorLookup import strength_array
from time import perf_counter
import numpy as np
import random

def canonicalize_hands(hands):
    """
    Canonicalize a batch of poker hands by reassigning suits to consistent values.
    Args:
        hands (np.ndarray): NumPy array of shape (n, 5, 2), where each hand is 5 cards,
                            each card is represented by (value, suit).
    Returns:
        np.ndarray: Canonicalized hands of the same shape.
    """
    # Extract values and suits
    values = hands[:, :, 0]  # Shape: (n, 5)
    suits = hands[:, :, 1]   # Shape: (n, 5)
    suits_canonical = np.zeros_like(suits)
    suits_canonical[:, 1] = np.where(suits[:, 0] == suits[:, 1], 0, 1)
    suits_canonical[:, 2] = np.where(
        suits[:, 0]==suits[:, 2], 0, np.where(
            suits[:,1]==suits[:,2], suits_canonical[:, 1], suits_canonical.max(axis=-1) + 1))
    suits_canonical[:, 3] = np.where(
        suits[:, 0] == suits[:, 3], 0, np.where(
            suits[:, 1] == suits[:, 3], suits_canonical[:, 1], np.where(
                suits[:, 2] == suits[:, 3], suits_canonical[:, 2], suits_canonical.max(axis=-1) + 1
            )
        )
    )
    suits_canonical[:, 4] = np.where(
        suits[:, 0] == suits[:, 4], 0, np.where(
            suits[:, 1] == suits[:, 4], suits_canonical[:, 1], np.where(
                suits[:, 2] == suits[:, 4], suits_canonical[:, 2], np.where(
                    suits[:, 3] == suits[:, 4], suits_canonical[:, 3], suits_canonical.max(axis=-1) + 1
                )
            )
        )
    )



    return np.stack([values, suits_canonical], axis=-1)


def generate_samples(remaining_deck, n):
    random.shuffle(remaining_deck)
    for _ in range(n):
        sample = random.sample(remaining_deck, 4)
        hand = [
            *sample[0],
            *sample[1],
            *sample[2],
            *sample[3],
        ]
        yield from hand


def generate_combinations(arr, r=2):
    """
    Generate all r-combinations of a 2D NumPy array efficiently using NumPy.
    This avoids Python-level loops and itertools.
    """
    n = len(arr)
    if r != 2:
        raise ValueError("This method currently supports only r=2 combinations.")

    # Generate indices for combinations
    i, j = np.triu_indices(n, k=1)  # Get upper triangle indices (no repeats, no self-pairs)

    # Use indices to select combinations
    comb_array = np.stack((arr[i], arr[j]), axis=1)
    return comb_array



hands = np.load("hero_flop.npy")

canonical_hands = canonicalize_hands(hands)

# Use np.unique to find unique rows and their mapping indices
_, indices, inverse = np.unique(canonical_hands.reshape(canonical_hands.shape[0], -1), axis=0, return_index=True, return_inverse=True)
unique_hands = canonical_hands[indices]

simple_deck = [(value, suit) for value in range(2, 15) for suit in range(4)]
results = []
n_sim = 5000

indices_runouts = np.array([np.random.choice(47, 4, replace=False) for _ in range(n_sim)])


for i in tqdm(range(unique_hands.shape[0])):

    hand_set = set(map(tuple, unique_hands[i]))  # Convert the hand to a set of tuples

    remaining = [card for card in simple_deck if card not in hand_set]

    # Generate runouts
    random.shuffle(remaining)
    runouts = np.array(remaining)[indices_runouts]

    hole_flop = np.tile(unique_hands[i], (runouts.shape[0], 1, 1))

    # Prepare hero strengths
    turn_river_array = runouts[:, 0:2, :]
    hero_array = np.hstack((hole_flop, turn_river_array))


    # Prepare villain strengths
    flop_array = hole_flop[:, 2:5, :]
    villain_array = np.hstack((flop_array, runouts))


    # Compute strengths
    strengths_hero = strength_array(hero_array)
    strengths_villain = strength_array(villain_array)

    p_win = (strengths_hero > strengths_villain).mean()
    p_loss = (strengths_hero < strengths_villain).mean()
    p_tie = (strengths_hero == strengths_villain).mean()
    results.append([p_tie, p_win, p_loss])
all_probs = np.array(results)[inverse]
np.save("flop_probs.npy", all_probs)
# Map results back to original hands
all_probs = np.array(results)[inverse]
np.save("flop_probs.npy", all_probs)


