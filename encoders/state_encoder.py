from encoders.card_encoder import encode_holecards, encode_flop, encode_card
from PokerGame.NLHoldem import Game, Player
from PokerGame.HandComperator import strength
from typing import List
import numpy as np

def card_vs_flop_suits(flop, card, encoded_flop_color):
    """encodes suit comparison of a card with flop"""
    card_color = np.zeros(2)
    if (encoded_flop_color == 1).all():  # one colored board 0 1 rainbow 1 0 hit
        if card.suit == flop[0].suit:
            card_color[0] = 1
        else:
            card_color[1] = 1
    elif encoded_flop_color[-1] == 1:  # two colored
        unique_suits, counts = np.unique([card.suit for card in flop], return_counts=True)
        turn_suit_count = counts[unique_suits == card.suit]
        if turn_suit_count.shape[0] == 0:
            card_color[0] = 1
        elif turn_suit_count == counts.max():
            card_color[:] = 1
        else:
            card_color[1] = 1
    else:  # rainbow 0 1 for continued rainbow and 1 0 for hit
        if np.unique([card.suit for card in flop + [card]]).shape[0] == 4:
            card_color[1] = 1
        else:
            card_color[0] = 1
    return card_color

def encode_board(board) -> np.array:
    turn_color = np.zeros(2)
    river_color = np.zeros(3)
    if board is not None:
        flop = encode_flop(board[:3])
        if len(board) == 4:
            turn = encode_card(board[-1])
            turn_color = card_vs_flop_suits(board[:-1], board[-1], flop[-2:])
            river = np.zeros(17)
        elif len(board) == 5 :
            turn = encode_card(board[-2])
            river = encode_card(board[-1])
            turn_color = card_vs_flop_suits(board[:-2], board[-2], flop[-2:])
            # check if river and turn have same suit
            river_color[2] = int(board[-1].suit == board[-2].suit)
            # encode river vs flop suits
            river_color[:-1] = card_vs_flop_suits(board[:-1], board[-1], flop[-2:])
        else:
            turn = np.zeros(17)
            river = np.zeros(17)
        boardcards = np.hstack([flop, turn, river])
    else:
        boardcards = np.zeros(65)
    return np.hstack([boardcards, turn_color, river_color])

def encode_state(game: Game, player:Player, n_players=None) -> np.array:
    street = np.zeros(4)
    street[game.street] = 1
    n_encode = game.n_players if n_players is None else n_players
    position = np.zeros(n_encode) # to allow learning agent for up to n-players
    idx_player = game.positions.index(player)
    position[idx_player] = 1
    starting_stacks = [player.starting_stack for player in game.positions]
    in_play = [int(player in game.left_in_hand) for player in game.positions]
    stacks = [player.stack for player in game.positions]
    bets = [player.bet for player in game.positions]

    last_action = np.zeros(n_encode*2)
    if game.last_action is not None:
        idx_last = game.positions.index(game.last_action["player"])
        last_action[idx_last] = game.last_action["action"]
        amount = game.last_action["amount"]
        if amount is None:
            amount = 0
        last_action[idx_last + n_encode] = amount

    holecards = encode_holecards(player.holecards)
    board = encode_board(game.board)
    legal_actions_array = np.zeros(3)
    if player == game.acting_player:
        legal_actions = list(game.get_legal_actions())
        legal_actions_array[legal_actions] = 1
    state = np.hstack([
        position,
        street,
        holecards,
        board,
        in_play,
        player.stack,
        player.bet,
        stacks,
        bets,
        game.n_players,
        starting_stacks,
        player.starting_stack,
        legal_actions_array
    ])
    return state


if __name__ == "__main__":
    stacks = [100, 25]
    while True:
        game = Game(n_players=2, stacks=stacks, hand_history=True)
        game.new_hand(first_hand=True)
        cards = game.players[0].holecards
        hole_encoded = encode_holecards(cards)
        state = encode_state(game, game.acting_player)
        game.implement_action(game.acting_player, 1)
        game.implement_action(game.acting_player, 1)
        flop = game.board
        state = encode_state(game, game.acting_player)
        game.implement_action(game.acting_player, 1)
        game.implement_action(game.acting_player, 1)
        turn = game.board[-1]
        state = encode_state(game, game.acting_player)

        game.implement_action(game.acting_player, 1)
        game.implement_action(game.acting_player, 1)
        river = game.board[-1]
        state = encode_state(game, game.acting_player)
        pass
