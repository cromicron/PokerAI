import torch

from PokerGame.NLHoldem import Game, Player
from PokerGame.HandComperator import strength as evaluate_strength
from pokerAI.encoders.state_encoder import encode_state
from operator import attrgetter
from itertools import combinations
import numpy as np



def encode_strength(holecards, board = None, update=True):
    type = np.zeros(8)
    holecards_sorted = sorted(holecards, key=attrgetter("value", "suit"), reverse=True)
    board_size = len(board) if board is not None else 0
    straight_draw = np.zeros(3)
    flushdraw = 0
    if board_size == 0:
        substrength = holecards_sorted[0].value*0.01
        strength = int(holecards[0].value == holecards[1].value)
        if strength == 0:
            substrength += holecards_sorted[1].value * 0.0001
        strength_float = strength + substrength
        strength_board = 0
    else:
        #
        all_cards = list(holecards) + board
        all_cards_list = [(card.value, card.suit) for card in all_cards]
        if board_size < 5:
            all_cards_list += [(-7, -11)] # add meaningless cards to make strength funct work
        if board_size < 4:
            all_cards_list += [(-99, -9)]
        strength_float = evaluate_strength(all_cards_list)
        strength = int(strength_float)
        substrength = strength_float % 1

    # how much holecards contribute. Calc strength with and without holecards
        if board_size == 3:
            values = sorted([card.value for card in board])
            strength_board = np.unique(np.array(values), return_counts=True)[1].max() - 1
            if strength_board == 2:
                strength_board += 0.01*values[0]
            elif strength_board == 1:
                strength_board += 0.01*values[-1] + 0.0001*values[0]
            else:
                strength_board += 1e-2*values[-1] + 1e-4*values[-2] + 1e-6*values[0]


        elif board_size == 4:
            board_cards_list = [(card.value, card.suit) for card in board]
            board_cards_list += [(-23, -177), (-11, -76), (-1237, -211)]
            strength_board = evaluate_strength(board_cards_list)
        else:
            # full board
            board_cards_list = [(card.value, card.suit) for card in board]
            board_cards_list += [(-23, -177), (-11, -76)]
            strength_board = evaluate_strength(board_cards_list)

        # check flush and straight draw
        if board_size < 5:
            suits = [card.suit for card in board]
            if strength <4: # check straight draw

                ranks_unique = set([card[0] for card in all_cards_list[:2+board_size]])
                combos = np.array(list(combinations(sorted(ranks_unique), 4)))

                if combos.shape[0] != 0:
                    if 14 in ranks_unique:
                        # Create a mask to find rows containing the value to replace
                        mask = combos == 14
                        ace_to_one = np.where(mask, 1, combos)
                        rows_with_ones = np.sort(ace_to_one[mask.any(axis=1)], axis=-1)  # Select rows with replacements
                        combos = np.vstack([combos, rows_with_ones])
                    combos_diff = combos[:, -1] - combos[:, 0]
                    open_ended = combos[combos_diff == 3]
                    gutshot = combos[combos_diff == 4]
                    if gutshot.shape[0] > 0:
                        sum_total = (5 * (gutshot[:, 0] + gutshot[:, -1])) // 2
                        sum_actual = np.sum(gutshot, axis=1)
                        missing_cards = set(np.unique((sum_total - sum_actual)))
                        if len(missing_cards) == 1:
                            straight_draw[0] = 1
                        else:
                            straight_draw[1] = 1
                    if open_ended.shape[0] != 0:
                        straight_draw[2] = 1
            if strength < 5:
                if holecards[0].suit == holecards[1].suit:
                    board_suit_count = suits.count(holecards[0].suit)
                    flushdraw = 1 if board_suit_count == 2 else 0
                else:
                    board_suit_count_0 = suits.count(holecards[0].suit)
                    board_suit_count_1 = suits.count(holecards[1].suit)
                    flushdraw = 1 if (
                            (board_suit_count_0 == 3) or (board_suit_count_1 == 3)
                    ) else 0

    if strength > 0:
        type[strength - 1] = 1
    strength_diff = strength_float - strength_board
    return np.hstack([type, strength_float, substrength, strength_diff, flushdraw, straight_draw])


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