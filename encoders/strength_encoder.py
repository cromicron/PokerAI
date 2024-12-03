from PokerGame.NLHoldem import Game, Player
from PokerGame.HandComperator import strength as evaluate_strength
from encoders.state_encoder import encode_state
import numpy as np
from operator import attrgetter

def encode_strength(holecards, board = None):
    type = np.zeros(8)
    holecards_sorted = sorted(holecards, key=attrgetter("value", "suit"), reverse=True)
    if board is None or len(board) == 0:
        substrength = holecards_sorted[0].value*0.01
        strength = int(holecards[0].value == holecards[1].value)
        if strength == 0:
            substrength += holecards_sorted[1].value * 0.0001
        strength_float = strength + substrength
        strength_board = 0
        flushdraw = 0
    else:
        all_cards = holecards + board
        all_cards_list = [(card.value, card.suit) for card in all_cards]
        if len(board) < 5:
            all_cards_list += [(-7, -11)] # add meaningless cards to make strength funct work
        if len(board) < 4:
            all_cards_list += [(-99, -9)]
        strength_float = evaluate_strength(all_cards_list)
        strength = int(strength_float)
        substrength = strength_float % 1

    # how much holecards contribute. Calc strength with and without holecards
        if len(board) == 3:
            values = sorted([card.value for card in board])
            strength_board = np.unique(np.array(values), return_counts=True)[1].max() - 1
            if strength_board == 2:
                strength_board += 0.01*values[0]
            elif strength_board == 1:
                strength_board += 0.01*values[-1] + 0.0001*values[0]
            else:
                strength_board += 1e-2*values[-1] + 1e-4*values[-2] + 1e-6*values[0]


        elif len(board) == 4:
            board_cards_list = [(card.value, card.suit) for card in board]
            board_cards_list += [(-23, -177), (-11, -76), (-1237, -211)]
            strength_board = evaluate_strength(board_cards_list)
        else:
            # full board
            board_cards_list = [(card.value, card.suit) for card in board]
            board_cards_list += [(-23, -177), (-11, -76)]
            strength_board = evaluate_strength(board_cards_list)

        # check flush-draw
        if len(board) < 5:
            suits = [card.suit for card in board]
            if holecards[0].suit == holecards[1].suit:
                board_suit_count = suits.count(holecards[0].suit)
                flushdraw = 1 if board_suit_count == 2 else 0
            else:
                board_suit_count_0 = suits.count(holecards[0].suit)
                board_suit_count_1 = suits.count(holecards[1].suit)
                flushdraw = 1 if (
                        (board_suit_count_0 == 3) or (board_suit_count_1 == 3)
                ) else 0
        else:
            flushdraw = 0
    if strength > 0:
        type[strength - 1] = 1
    strength_diff = strength_float - strength_board
    return np.hstack([type, strength_float, substrength, strength_diff, flushdraw])


if __name__ == "__main__":
    stacks = [100, 25]
    while True:
        game = Game(n_players=2, stacks=stacks, hand_history=True)
        game.new_hand(first_hand=True)
        cards = game.players[0].holecards
        value = encode_strength(cards)
        state = encode_state(game, game.acting_player)
        game.implement_action(game.acting_player, 1)
        game.implement_action(game.acting_player, 1)
        flop = game.board
        state = encode_state(game, game.acting_player)
        value = encode_strength(cards, flop)
        game.implement_action(game.acting_player, 1)
        game.implement_action(game.acting_player, 1)
        turn = game.board[-1]
        state = encode_state(game, game.acting_player)
        value = encode_strength(cards, game.board)
        game.implement_action(game.acting_player, 1)
        game.implement_action(game.acting_player, 1)
        river = game.board[-1]
        state = encode_state(game, game.acting_player)
        value = encode_strength(cards, game.board)
        pass