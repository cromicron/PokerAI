from encoders.hand_encoder import PokerHandEmbedding
import torch
from PokerGame.NLHoldem import Card
import numpy as np
from itertools import combinations
from lookup.HandComparatorLookup import strength_array, strength

deck = [(rank, suit) for rank in range(2, 15) for suit in range(4)]
card_to_int = {
    card: i for i, card in enumerate(deck)
}
combo_indices = np.array(list(combinations(range(45), 2)))
def encode_cards(holecards, board=None):
    preflop = [card_to_int[(card.value, card.suit)] for card in holecards]
    if board:
        board_encoded =  [card_to_int[(card.value, card.suit)] for card in board]
    else:
        board_encoded = None
    return preflop, board_encoded
def format_hand(holecards, board=None):
    suits = ['♠', '♥', '♦', '♣']  # 0 to 3 suits
    rank_map = {i: str(i) for i in range(2, 11)}
    rank_map.update({11: 'J', 12: 'Q', 13: 'K', 14: 'A'})

    holecards_str = f"[{rank_map[holecards[0].value]}{suits[holecards[0].suit]}, {rank_map[holecards[1].value]}{suits[holecards[1].suit]}]"
    board_str = ', '.join(f"{rank_map[card.value]}{suits[card.suit]}" for card in board) if board else "No board"
    return f"Hand: {holecards_str}, Board: [{board_str}]"

feature_dim = 256  # Size of output feature vectors
deep_layer_dims = (512, 2048, 2048, 2048)
intermediary_dim = 16
embedding_dim = 4
model = PokerHandEmbedding(embedding_dim, feature_dim, deep_layer_dims, intermediary_dim)
model.load_state_dict(torch.load("model_checkpoint.pth")["model_state_dict"], strict=False)

# Test cases
categories = {
    "Normal Hands":
[
        ([Card(14, 3), Card(14, 0)], [Card(12, 2), Card(13, 2), Card(12, 1), Card(11, 2), Card(2, 2)]),
        ([Card(4, 1), Card(4, 2)], [Card(14, 3), Card(14, 2), Card(14, 1), Card(13, 0), Card(3, 3)]),
        ([Card(10, 1), Card(4, 2)], [Card(10, 3), Card(9, 3), Card(8, 3), Card(7, 3), Card(3, 0)]),
        ([Card(10, 1), Card(8, 2)], [Card(11, 3), Card(8, 3), Card(2, 0), Card(7, 3), Card(3, 0)]),
    ],

    "Split Pot Certainty" : [
        ([Card(12, 3), Card(12, 0)], [Card(14, 2), Card(13, 2), Card(12, 2), Card(11, 2), Card(10, 2)]),  # Straight flush
        ([Card(4, 1), Card(4, 2)], [Card(14, 3), Card(14, 2), Card(14, 1), Card(14, 0), Card(13, 3)]), # Open-ended straight draw
    ],

    "Complex River Scenarios": [
        ([Card(10, 2), Card(9, 2)], [Card(6, 2), Card(7, 2), Card(8, 2), Card(4, 1), Card(5, 2)]),  # Straight flush
        ([Card(4, 1), Card(4, 2)], [Card(4, 3), Card(9, 1), Card(10, 1), Card(4, 0), Card(5, 3)]),  # Quads
    ],
    "Trashy Hands That Look Strong": [
        ([Card(8, 1), Card(7, 3)], [Card(6, 2), Card(5, 2), Card(4, 2), Card(3, 2), Card(2, 2)]),  # Counterfeited straight
        ([Card(2, 1), Card(2, 2)], [Card(2, 3), Card(8, 2), Card(8, 3), Card(8, 1), Card(8, 0)]),  # Weak trips
    ],
}

# Encode and evaluate
tests = []
for category, hands_boards in categories.items():
    for holecards, board in hands_boards:
        encoded = encode_cards(holecards, board)
        holecards_simple = [(card.value, card.suit) for card in holecards]
        board_simple = [(card.value, card.suit) for card in board]
        remaining_deck = set(deck) - set(holecards_simple) - set(board_simple)
        villain_holecards = np.array(list(remaining_deck))[combo_indices]
        villain_hands = np.hstack([villain_holecards, np.tile(board_simple,  (990, 1, 1))])
        strength_hero = strength(holecards_simple+board_simple)
        strengths_villain = strength_array(villain_hands)
        p_tie = (strength_hero == strengths_villain).mean()
        p_win = (strength_hero > strengths_villain).mean()
        p_loose = (strength_hero < strengths_villain).mean()
        tests.append((category, encoded, format_hand(holecards, board), (p_tie, p_win, p_loose)))

# Convert to tensor and predict
predictions = []
for test in tests:
    hands = test[1]
    preflop_tensor = torch.tensor(hands[0], dtype=torch.long).unsqueeze(0)
    if len(hands) > 1:
        cards = hands[1]
        flop_tensor = torch.tensor(cards[:3], dtype=torch.long).unsqueeze(0)

        turn_tensor = torch.tensor(cards[3:4], dtype=torch.long).unsqueeze(0)

        river_tensor = torch.tensor(cards[4:], dtype=torch.long).unsqueeze(0)
    else:
        flop_tensor = None
        turn_tensor = None
        river_tensor = None

    features, model_predictions = model(preflop_tensor, flop_tensor, turn_tensor, river_tensor)
    predictions.append(torch.exp(model_predictions["log_probs_outcome_river"]))


# Print results
print("\nResults:\n" + "=" * 50)
current_category = None
for i, (category, _, description, truth) in enumerate(tests):
    if category != current_category:
        print(f"\n{category}:\n" + "-" * 50)
        current_category = category
    print(f"{description}\nPrediction (Tie/Win/Loss): {predictions[i][0].tolist()}")
    print(f"True (Tie/Win/Loss): {truth}\n" + "-" * 50)
