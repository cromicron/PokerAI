import tqdm
import numpy as np
import random

import tqdm
from treys import Card as TreysCard
from treys import Evaluator
from PokerGame.NLHoldem import Deck
from PokerGame.HandComperator import compare_hands
# Your Card and Deck class definitions go here.

# Your poker hand checking functions (check_straightflush, check_quads, etc.) and compare_hands go here.

# Simulate a few poker hands
def simulate_hands(num_hands, hand_size):
    deck = Deck()
    deck.shuffle()
    return [random.sample(deck, hand_size) for _ in range(num_hands)]

# Convert hands to the format required by your evaluation functions
def convert_to_hand_format(hands):
    return [[(card.value, card.suit) for card in hand] for hand in hands]

import numpy as np

def compare_sort_order(list1, list2):
    # Get the indices that would sort the arrays
    sorted_indices_list1 = np.argsort(list1)
    sorted_indices_list2 = np.argsort(list2)

    # Check if the sorted indices are the same
    # Since argsort can give different indices for same-value elements, we rank them and then compare ranks.
    rank_list1 = np.argsort(sorted_indices_list1)
    rank_list2 = np.argsort(sorted_indices_list2)

    # Compare the ranks
    return np.array_equal(rank_list1, rank_list2)


def compare_hands_trey(converted_hands):
    treys_results = []
    for hand in converted_hands:
        # Convert the hand to a treys Card format
        treys_hand = [TreysCard.new(card.representation) for card in simulated_hands[converted_hands.index(hand)]]
        # Evaluate with treys (assuming no community cards for simplicity)
        score = evaluator.evaluate([], treys_hand)
        treys_results.append(-score)
    return treys_results

# Set up treys Evaluator
evaluator = Evaluator()

n_hands = 1000000
for i in tqdm.tqdm(range(n_hands)):
    n_incorrect = 0
    # Simulate hands
    num_hands = random.randint(2,7)
    hand_size = 7
    simulated_hands = simulate_hands(num_hands, hand_size)
    converted_hands = convert_to_hand_format(simulated_hands)

    # Use your compare_hands function to evaluate these hands
    my_game_results = compare_hands(simulated_hands)

    # Convert the hands to the treys format and evaluate them using the treys Evaluator


    treys_results = compare_hands_trey(converted_hands)
    if not compare_sort_order(treys_results, my_game_results):
        n_incorrect += 1
        print(converted_hands)
print(f"{n_incorrect} hands have different scorings")
