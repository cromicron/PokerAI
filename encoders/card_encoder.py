from PokerGame.NLHoldem import Game, Card
from typing import List
import numpy as np
from operator import attrgetter


def encode_holecards(cards: List[Card]) -> np.array:
    cards_sorted = sorted(cards, key=attrgetter("value", "suit"))
    values = np.zeros(13)
    indices_val = [card.value - 2 for card in cards_sorted]
    values[indices_val] = 1
    indices_suit = [cards_sorted[0].suit, 4+ cards_sorted[1].suit]
    suits = np.zeros(8)
    suits[indices_suit] = 1
    pair = int(values.sum() == 1) # redundant, but might help in learning
    suited = int(cards_sorted[0].suit == cards_sorted[1].suit)
    return np.hstack([values, suits, pair, suited])

def encode_flop(cards: List[Card]) -> np.array:
    cards_sorted = sorted(cards, key=attrgetter("value", "suit"))
    values = np.zeros(13)
    indices_val = [card.value - 2 for card in cards_sorted]
    values[indices_val] = 1
    suits = np.zeros(12)
    indices_suit = [cards_sorted[0].suit, 4+ cards_sorted[1].suit, 8+cards_sorted[2].suit]
    suits[indices_suit] = 1
    # encode colors 01 rainbow 10 two same color 11 all same
    colors = np.zeros(2)
    n_suit_max = suits.reshape(3, 4).sum(axis=0).max()
    if n_suit_max == 1: # rainbow
        colors[0] = 1
    elif n_suit_max == 2: # two suits same
        colors[1] = 1
    else: # all same suit
        colors[:] = 1

    # first and second same
    trips = int(values.sum() == 1)
    pair = int(values.sum() == 2)
    if trips:
        same_0 = same_1 = 1
    elif pair:
        same_0 = int(cards_sorted[0].value == cards_sorted[1].value) # middle and first equal value
        same_1 = int(cards_sorted[1].value == cards_sorted[2].value)
        pass# middle and last equal value
    else:
        same_0 = same_1 = 0


    return np.hstack([values, suits, trips, pair, same_0, same_1, colors])

def encode_card(card: Card) -> np.array:
    value = np.zeros(13)
    suit = np.zeros(4)
    value[card.value-2] = 1
    suit[card.suit] = 1
    return np.hstack([value, suit])




if __name__ == "__main__":
    stacks = [100, 25]
    while True:
        game = Game(n_players=2, stacks=stacks, hand_history=True)
        game.new_hand(first_hand=True)
        cards = game.players[0].holecards
        hole_encoded = encode_holecards(cards)
        game.implement_action(game.next[-1], 1)
        game.implement_action(game.next[-1], 1)
        flop = game.board
        flop_encoded = encode_flop(flop)
        game.implement_action(game.next[-1], 1)
        game.implement_action(game.next[-1], 1)
        turn = game.board[-1]
        turn_encoded = encode_card(turn)

        game.implement_action(game.next[-1], 1)
        game.implement_action(game.next[-1], 1)
        river = game.board[-1]
        river_encoded = encode_card(turn)


