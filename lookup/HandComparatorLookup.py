from PokerGame.NLHoldem import Deck
import random
import pickle
import numpy as np
import os


flush_lookup_path = os.path.join(os.path.dirname(__file__), "flush_lookup.pkl")
int_flush_lookup_path = os.path.join(os.path.dirname(__file__), "int_flush_lookup.pkl")
strength_lookup_path = os.path.join(os.path.dirname(__file__), "lookup.pkl")

with open(flush_lookup_path, "rb") as f:
    suit_dict = pickle.load(f)
with open(int_flush_lookup_path, "rb") as f:
    int_suit_dict = pickle.load(f)
with open(strength_lookup_path, "rb") as f:
    strength_dict = pickle.load(f)

suit_weights = 7 ** np.arange(6, -1, -1)
suit_keys = np.array(sorted(int_suit_dict.keys()))  # Sorted keys
suit_values = np.array([int_suit_dict[k] for k in suit_keys])  # Corresponding values

strength_keys = np.array(sorted(strength_dict.keys()))  # Sorted keys
strength_values = np.array([strength_dict[k] for k in strength_keys])  # Corresponding values



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

def strength(hand):
    hand = hand[:]
    hand.sort(reverse=True)
    suits = tuple([card[1] for card in hand])
    ranks = [card[0] for card in hand]
    encoded_hand = encode_hand(ranks, suit_dict[suits])
    return strength_dict[encoded_hand]




def strength_array(hands, sorted=False):

    # Extract ranks and suits
    ranks = hands[..., 0]  # Shape: (n, 7)
    suits = hands[..., 1]  # Shape: (n, 7)

    # Sort ranks and retrieve sorting indices
    if not sorted:
        sorted_indices = np.argsort(-ranks, axis=-1)

        # Directly reorder ranks and suits using advanced indexing
        batch_indices = np.arange(hands.shape[0])[:, None]
        sorted_ranks = ranks[batch_indices, sorted_indices]
        sorted_suits = suits[batch_indices, sorted_indices]
    else:
        sorted_ranks = ranks
        sorted_suits = suits
    # Encode suits
    suit_ints = sorted_suits @ suit_weights
    suit_indices = np.searchsorted(suit_keys, suit_ints)
    try:
        suits_encoded = suit_values[suit_indices]
    except:
        print("well")

    # Encode hands
    hand_encoded = encode_hands_array(sorted_ranks, suits_encoded)

    # Lookup hand strengths
    hand_strength_indices = np.searchsorted(strength_keys, hand_encoded)
    hand_strengths = strength_values[hand_strength_indices]



    return hand_strengths



def compare_hands(hands: list):
    # Takes in lists of seven poker cards and returns the value of each hand
    strengths = [strength(hand) for hand in hands]
    return strengths


if __name__ == "__main__":
    deck = Deck()
    hands = []
    for i in range(100000):
        hand = random.sample(deck, 7)
        hand_array = [[card.value, card.suit] for card in hand]
        strength(hand_array)
        hands.append(hand_array)
    hands_array = np.stack(hands)
    strength_array(hands_array)

