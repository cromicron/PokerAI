from PokerGame.NLHoldem import Deck
from itertools import combinations
import random
from tqdm import tqdm
import numpy as np
from math import comb
import pickle
from lookup.HandComparatorLookup import strength_array

# Function to generate unique hole cards
def generate_unique_hole_cards(deck):
    unique_hole_cards = {}
    for i, card1 in enumerate(deck):
        for j, card2 in enumerate(deck):
            if i >= j:  # Avoid duplicates and self-pairs
                continue

            # Sort cards by value
            if card1.value > card2.value:
                higher_card, lower_card = card1, card2
            else:
                higher_card, lower_card = card2, card1

            # Determine the "type" of the hole card
            if higher_card.value == lower_card.value:  # Pair (e.g., AA)
                hand_type = (higher_card.value, "pair")
            else:
                suited = higher_card.suit == lower_card.suit
                hand_type = (higher_card.value, lower_card.value, "suited" if suited else "off-suited")

            # Store the first instance of each type
            if hand_type not in unique_hole_cards:
                unique_hole_cards[hand_type] = (higher_card, lower_card)

    return list(unique_hole_cards.values())


def generate_combinations(deck_size, choose):
    """
    Generate all combinations of cards from a deck as a NumPy array.

    Args:
    - deck_size: Total number of cards in the deck (e.g., 50).
    - choose: Number of cards to choose (e.g., 7).

    Returns:
    - combinations_array: NumPy array of shape (n_combinations, choose).
    """
    # Total number of combinations
    n_combinations = comb(deck_size, choose)

    # Preallocate NumPy array for all combinations
    combinations_array = np.empty((n_combinations, choose), dtype=np.int8)

    # Fill the array with combinations
    for idx, combination in enumerate(combinations(range(deck_size), choose)):
        combinations_array[idx] = combination

    return combinations_array

def generate_samples():
    for _ in range(n_monte_carlo):
        for card in random.sample(remaining_cards, 7):
            yield card.value
            yield card.suit  # suit

n_monte_carlo = 10000000

if __name__ == "__main__":
    with open("preflop_probs.pkl", "rb") as f:
        d = pickle.load(f)
    deck = Deck()
    results = {}
    unique_hole_cards = generate_unique_hole_cards(deck)

    for hole in tqdm(unique_hole_cards):
        remaining_cards = set(deck) - set(hole)
        samples = np.fromiter(generate_samples(), dtype=int).reshape((n_monte_carlo, 7, 2))
        hole_array = np.tile([(card.value, card.suit) for card in hole], (n_monte_carlo, 1, 1))
        hero = np.concatenate([hole_array, samples[:, :-2, :]], axis=1)
        strengths_hero = strength_array(hero)
        strengths_villain = strength_array(samples)
        # Count smaller and larger comparisons using searchsorted


        # Calculate win (Hero wins when hero > villain)
        p_win = (strengths_hero > strengths_villain).mean()
        p_loss = (strengths_hero < strengths_villain).mean()
        p_tie = (strengths_hero == strengths_villain).mean()

        if hole[0].value == hole[1].value:
            hole_repr = (hole[0].value, hole[0].value, 0)
        elif hole[0].suit == hole[1].suit:
            hole_repr = (hole[0].value, hole[1].value, 1)
        else:
            hole_repr = (hole[0].value, hole[1].value, 0)
        results[hole_repr] = [p_tie, p_win, p_loss]

    with open("preflop_probs.pkl", "wb") as f:
        pickle.dump(results, f)
