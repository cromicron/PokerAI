import pickle
import os


def encode_hand(hand):
    """
    Encodes a list of 5 two-tuples into a single integer.

    :param tuples: List of 5 tuples where each tuple contains:
                   - First element: int between 2 and 14 (inclusive)
                   - Second element: int between 0 and 3 (inclusive)
    :return: Encoded integer
    """
    suits = [card[1] for card in hand]
    suits_encoded = lookup_suits[tuple(suits)]
    hand_suit_enc = [(hand[i][0], suits_encoded[i])for i in range(5)]
    encoded = 0
    for i, (first, second) in enumerate(hand_suit_enc):
        # Shift the encoded integer to make space for the next tuple
        encoded <<= 6  # Each tuple takes 6 bits
        # Pack the first and second elements into the integer
        encoded |= ((first - 2) << 2) | second  # Use 4 bits for first, 2 bits for second
    return encoded

lookup_probs_path = os.path.join(os.path.dirname(__file__), "lookup_probs_flop.pkl")
lookup_suits_path = os.path.join(os.path.dirname(__file__), "flop_suits_lookup.pkl")

with open(lookup_probs_path, "rb") as f:
    lookup= pickle.load(f)

with open(lookup_suits_path, "rb") as f:
    lookup_suits= pickle.load(f)

def get_probs_flop(hand, is_sorted=False):
    if not is_sorted:
        hand = hand[:]
        hand = sorted(hand[:2]) + sorted(hand[2:])
    hand_encoded = encode_hand(hand)
    probs = lookup[hand_encoded]
    return probs

if __name__ == "__main__":
    hand = [(4,0), (11,1), (3,3), (5, 1), (14,2)]
    probs = get_probs_flop(hand)
    print(probs)