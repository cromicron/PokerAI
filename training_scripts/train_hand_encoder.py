import random
from multiprocessing import Process, Queue, Lock, set_start_method, Event
from collections import Counter
import torch
import torch_optimizer as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from itertools import combinations
import numpy as np
from encoders.hand_encoder import PokerHandEmbedding
from lookup.HandComparatorLookup import  strength, strength_array
from lookup.lookup_probs_flop import get_probs_flop
from PokerGame.HandComperator import strength as strength_old

import pickle


def log_ratio_loss(p, q, eps=1e-4):
    # Add a small epsilon to prevent log(0)
    loss = torch.abs(torch.log(p.clamp(min=eps)) -torch.clamp(q, min=np.log(eps))).sum(dim=-1)
    return loss

with open("../lookup/preflop_probs.pkl", "rb") as f:
    LOOKUP_PREFLOP = pickle.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deck = [(rank, suit) for rank in range(2, 15) for suit in range(4)]
card_to_int = {
    card: i for i, card in enumerate(deck)
}

in_to_card = {
    i: card for i, card in enumerate(deck)
}
COMBO_INDICES_RIVER = np.array(list(combinations(range(45), 2)))
N_SIM_TURN = 250
COMBO_INDICES_TURN = np.stack([np.random.choice(range(46), 3, replace=False) for _ in range(N_SIM_TURN)])
MENAINGLESS_CARDS = [(-7, -11), (-99, -9), (-321, -77)]

class PredictionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



def encode_straight_draw(hand):
    # check flush and straight draw
    ranks_unique = set([card[0] for card in hand])
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
        if open_ended.shape[0] != 0:
            return 2
        if gutshot.shape[0] > 0:
            sum_total = (5 * (gutshot[:, 0] + gutshot[:, -1])) // 2
            sum_actual = np.sum(gutshot, axis=1)
            missing_cards = set(np.unique((sum_total - sum_actual)))
            if len(missing_cards) == 1:
                return 1
            else:
                return 0

    return 0

def encode_flush_draws(hand):
    suits = [card[1] for card in hand]
    if hand[0][1] == hand[1][1]:
        board_suit_count = suits.count(hand[0][1])
        flushdraw = 1 if board_suit_count == 2 else 0
    else:
        board_suit_count_0 = suits.count(hand[0][1])
        board_suit_count_1 = suits.count(hand[0][1])
        flushdraw = 1 if (
                (board_suit_count_0 == 3) or (board_suit_count_1 == 3)
        ) else 0
    return flushdraw

def board_strengths(board):
    # flop
    ranks = [card[0] for card in board[:3]]
    counter = Counter(ranks)
    max_count = counter.most_common(1)[0][1]
    strength_flop = max_count - 1
    # turn
    ranks = sorted([card[0] for card in board[:4]], reverse=True)
    counter = Counter(ranks)
    most_comon = counter.most_common(1)[0]
    max_count = most_comon[1]
    least_comon = counter.most_common()[-1]
    min_count = least_comon[1]
    if min_count == 2:
        strength_turn = 2
    elif max_count == 3:
        strength_turn = 3
    else:
        strength_turn = max_count -1

    if strength_turn == 0:
        strength_turn = 1e-2*ranks[0] + 1e-4*ranks[1] + 1e-6*ranks[2] + 1e-8*ranks[3]
    elif strength_turn == 1:
        pair = most_comon[0]
        kickers = [rank for rank in ranks if rank != pair]
        strength_turn += 1e-2*pair + 1e-4*kickers[0] + 1e-6*kickers[1]
    elif strength_turn == 2:
        pairs = sorted(list(set(ranks)))
        strength_turn += 1e-2*pairs[1] + 1e-4*pairs[0]
    elif strength_turn == 3:
        strength_turn += 1e-2* most_comon[0] + 1e-4*least_comon[0]
    else:
        strength_turn += 1e-2*ranks[0]



    strength_river = strength_old(board + MENAINGLESS_CARDS[:-1])

    return strength_flop, strength_turn, strength_river


def float_to_list(num, turn=False):
    # Ensure precision and convert to string
    num_str = f"{num:.8f}" if turn else f"{num:.10f}"
    integer_part, fractional_part = num_str.split('.')

    # Convert integer part and create the list of pairs from fractional part
    result = [int(integer_part)]
    result.extend(int(fractional_part[i:i + 2])-2 for i in range(0, len(fractional_part), 2))

    return result
def producer(buffer, chunk_size):
    """
    Producer adds large shuffled chunks of hands to the buffer and periodically shuffles the entire buffer.
    """

    while True:
        card_int_list = []
        hero_turn_list = []
        strengths_hero_list = []
        villain_hands_list = []
        hand_types_preflop = []
        hand_types_flop = []
        hand_types_turn = []
        hand_types_river = []
        probs_preflop = []
        straight_draws_flop = []
        straight_draws_turn = []
        flush_draws_flop = []
        flush_draws_turn = []

        board_strengths_flop = []
        board_strengths_turn = []
        board_strengths_river = []

        probs_flop = []
        runouts_turn = []

        hand_values_flop = []
        hand_values_turn = []
        hand_values_river = []

        hand_values_turn_board = []
        hand_values_river_board = []

        for _ in range(chunk_size):
            special_hands = random.random() < .2
            random.shuffle(deck)
            cards = deck[: 7]
            if special_hands:
                while strength_old(cards) < 4:
                    random.shuffle(deck)
                    cards = deck[: 7]

            remaining_cards_river = np.array(deck[7: ])

            cards_int = [card_to_int[card] for card in cards]
            # add special hand types

            strength_hero = strength(cards)

            if cards[0][0]==cards[1][0]:
                hand_type_preflop = 2
            elif cards[0][1] == cards[1][1]:
                hand_type_preflop = 1
            else:
                hand_type_preflop = 0
            hand_types_preflop.append(hand_type_preflop)
            strength_flop = strength_old(cards[:5] + MENAINGLESS_CARDS[:-1])
            strength_flop_cat = float_to_list(strength_flop)
            hand_types_flop.append(strength_flop_cat[0])
            hand_values_flop.append(strength_flop_cat)

            strength_turn = strength_old(cards[:6] + [MENAINGLESS_CARDS[0]])
            strength_turn_cat = float_to_list(strength_turn)
            hand_types_turn.append(strength_turn_cat[0])
            hand_values_turn.append(strength_turn_cat)

            strength_river = strength_old(cards)
            strength_river_cat = float_to_list(strength_river)
            hand_types_river.append(strength_river_cat[0])
            hand_values_river.append(strength_river_cat)

            hole_sorted = sorted(cards[:2], reverse=True)
            suited = 1 if hole_sorted[0][1] == hole_sorted[1][1] else 0
            cards_lookup = (hole_sorted[0][0], hole_sorted[1][0], suited)

            probs_preflop.append(LOOKUP_PREFLOP[cards_lookup])


            straight_draw_flop = encode_straight_draw(cards[:5]) if strength_flop < 4 else 0
            straight_draws_flop.append(straight_draw_flop)
            straight_draw_turn = encode_straight_draw(cards[:6]) if strength_turn < 4 else 0
            straight_draws_turn.append(straight_draw_turn)

            flush_draw_flop = encode_flush_draws(cards[:5]) if strength_flop < 5 else 0
            flush_draws_flop.append(flush_draw_flop)
            flush_draw_turn = encode_flush_draws(cards[:6]) if strength_turn < 5 else 0
            flush_draws_turn.append(flush_draw_turn)

            board_flop, board_turn, board_river = board_strengths(cards[2:])
            board_strengths_flop.append(board_flop)

            strength_turn_board_cat = float_to_list(board_turn, True)
            strength_river_board_cat = float_to_list(board_river)

            hand_values_turn_board.append(strength_turn_board_cat)
            board_strengths_turn.append(strength_turn_board_cat[0])
            hand_values_river_board.append(strength_river_board_cat)
            board_strengths_river.append(strength_river_board_cat[0])

            villain_holecards_river = remaining_cards_river[COMBO_INDICES_RIVER]
            villain_hands_river = np.hstack([villain_holecards_river, np.tile(cards[2:],  (990, 1, 1))])
            card_int_list.append(cards_int)
            strengths_hero_list.append(strength_hero)
            villain_hands_list.append(villain_hands_river)

            probs_flop.append(get_probs_flop(cards[:5]))

            remaining_cards_turn = np.array(deck[6:])
            turn_runout = remaining_cards_turn[COMBO_INDICES_TURN]
            hero_turn_list.append(cards[:6])
            runouts_turn.append(turn_runout)
        strengths_hero = np.array(strengths_hero_list)
        strengths_villain_flat = strength_array(np.vstack(villain_hands_list))
        # compare strengths
        strengths_villain = strengths_villain_flat.reshape(chunk_size, -1)
        p_tie = (strengths_hero[:, None] == strengths_villain).mean(axis=-1)
        p_win = (strengths_hero[:, None] > strengths_villain).mean(axis=-1)
        p_loss = (strengths_hero[:, None] < strengths_villain).mean(axis=-1)
        features = np.vstack(card_int_list)
        probs_river = np.column_stack([p_tie, p_win, p_loss])
        probs_flop = np.stack(probs_flop)

        # get probs for turn
        runouts_turn_array = np.stack(runouts_turn)
        hero_turn_array = np.array(hero_turn_list)
        hero_turn_array = np.broadcast_to(hero_turn_array[:, None, :, :], (chunk_size, N_SIM_TURN, 6, 2))
        hero_runouts = np.concatenate([hero_turn_array, runouts_turn_array[:, :, 0:1, :]], axis = 2)
        villain_turn = np.concatenate([hero_turn_array[:,:, 2:], runouts_turn_array], axis=2)
        all_turns = np.vstack([hero_runouts, villain_turn])
        turn_strengths = strength_array(all_turns.reshape(-1, 7, 2)).reshape(*all_turns.shape[:2])

        p_win_turn = (turn_strengths[:chunk_size] > turn_strengths[chunk_size:]).mean(axis=1)
        p_lose_turn = (turn_strengths[:chunk_size] < turn_strengths[chunk_size:]).mean(axis=1)
        p_tie_turn = (turn_strengths[:chunk_size] == turn_strengths[chunk_size:]).mean(axis=1)
        probs_turn = np.column_stack([p_tie_turn, p_win_turn, p_lose_turn])

        labels = {
            "outcome_preflop": np.stack(probs_preflop),
            "preflop_type": np.array(hand_types_preflop),

            "outcome_flop": probs_flop,
            "type_flop": np.array(hand_types_flop),
            "straight_draw_flop": np.array(straight_draws_flop),
            "flush_draw_flop": np.array(flush_draws_flop),
            "type_turn": np.array(hand_types_turn),
            "straight_draw_turn": np.array(straight_draws_turn),
            "flush_draw_turn":  np.array(flush_draws_turn),
            "outcome_turn": probs_turn,

            "outcome_river": probs_river,
            "type_river": np.array(hand_types_river),

            "board_strength_flop": np.array(board_strengths_flop),
            "board_strength_turn": np.array(board_strengths_turn),
            "board_strength_river": np.array(board_strengths_river),

            "hand_values_flop": np.array(hand_values_flop).astype(np.int64),
            "hand_values_turn": np.array(hand_values_turn).astype(np.int64),
            "hand_values_river": np.array(hand_values_river).astype(np.int64),

            "hand_values_turn_board": np.array(hand_values_turn_board).astype(np.int64),
            "hand_values_river_board": np.array(hand_values_river_board).astype(np.int64),
        }

        buffer.put([features, labels])


def calc_losses(predictions: dict, labels: dict):
    log_probs_preflop = predictions["log_probs_outcome_preflop"]
    log_probs_flop = predictions["log_probs_outcome_flop"]
    log_probs_turn = predictions["log_probs_outcome_turn"]
    log_probs_river = predictions["log_probs_outcome_river"]

    log_probs_type_preflop = predictions["log_probs_preflop_type"]
    log_probs_type_flop = predictions["log_probs_type_flop"]
    log_probs_type_turn = predictions["log_probs_type_turn"]
    log_probs_type_river = predictions["log_probs_type_river"]

    log_probs_straight_draw_flop = predictions["log_probs_straight_draw_flop"]
    log_probs_flush_draw_flop = predictions["log_probs_flush_draw_flop"]
    log_probs_straight_draw_turn = predictions["log_probs_straight_draw_turn"]
    log_probs_flush_draw_turn = predictions["log_probs_flush_draw_turn"]

    log_probs_board_strength_flop = predictions["log_probs_board_strength_flop"]
    log_probs_board_strength_turn = predictions["log_probs_board_strength_turn"]
    log_probs_board_strength_river = predictions["log_probs_board_strength_river"]

    log_probs_flop_kicker_0 = predictions["log_probs_flop_kicker_0"]
    log_probs_flop_kicker_1 = predictions["log_probs_flop_kicker_1"]
    log_probs_flop_kicker_2 = predictions["log_probs_flop_kicker_2"]
    log_probs_flop_kicker_3 = predictions["log_probs_flop_kicker_3"]
    log_probs_flop_kicker_4 = predictions["log_probs_flop_kicker_4"]

    log_probs_turn_kicker_0 = predictions["log_probs_turn_kicker_0"]
    log_probs_turn_kicker_1 = predictions["log_probs_turn_kicker_1"]
    log_probs_turn_kicker_2 = predictions["log_probs_turn_kicker_2"]
    log_probs_turn_kicker_3 = predictions["log_probs_turn_kicker_3"]
    log_probs_turn_kicker_4 = predictions["log_probs_turn_kicker_4"]

    log_probs_river_kicker_0 = predictions["log_probs_river_kicker_0"]
    log_probs_river_kicker_1 = predictions["log_probs_river_kicker_1"]
    log_probs_river_kicker_2 = predictions["log_probs_river_kicker_2"]
    log_probs_river_kicker_3 = predictions["log_probs_river_kicker_3"]
    log_probs_river_kicker_4 = predictions["log_probs_river_kicker_4"]

    log_probs_turn_board_kicker_0 = predictions["log_probs_turn_board_kicker_0"]
    log_probs_turn_board_kicker_1 = predictions["log_probs_turn_board_kicker_1"]
    log_probs_turn_board_kicker_2 = predictions["log_probs_turn_board_kicker_2"]
    log_probs_turn_board_kicker_3 = predictions["log_probs_turn_board_kicker_3"]

    log_probs_river_board_kicker_0 = predictions["log_probs_river_board_kicker_0"]
    log_probs_river_board_kicker_1 = predictions["log_probs_river_board_kicker_1"]
    log_probs_river_board_kicker_2 = predictions["log_probs_river_board_kicker_2"]
    log_probs_river_board_kicker_3 = predictions["log_probs_river_board_kicker_3"]
    log_probs_river_board_kicker_4 = predictions["log_probs_river_board_kicker_4"]



    y_outcome_preflop = torch.tensor(labels["outcome_preflop"], dtype=torch.float32, device=device)
    y_outcome_flop = torch.tensor(labels["outcome_flop"], dtype=torch.float32, device=device)
    y_outcome_turn = torch.tensor(labels["outcome_turn"], dtype=torch.float32, device=device)
    y_outcome_river = torch.tensor(labels["outcome_river"], dtype=torch.float32, device=device)
    y_type_preflop = torch.tensor(labels["preflop_type"], dtype=torch.long, device=device)
    y_type_flop = torch.tensor(labels["type_flop"], dtype=torch.long, device=device)
    y_type_turn = torch.tensor(labels["type_turn"], dtype=torch.long, device=device)
    y_type_river = torch.tensor(labels["type_river"], dtype=torch.long, device=device)

    y_straight_draw_flop = torch.tensor(labels["straight_draw_flop"], dtype=torch.long, device=device)
    y_flush_draw_flop = torch.tensor(labels["flush_draw_flop"], dtype=torch.long, device=device)
    y_straight_draw_turn = torch.tensor(labels["straight_draw_turn"], dtype=torch.long, device=device)
    y_flush_draw_turn = torch.tensor(labels["flush_draw_turn"], dtype=torch.long, device=device)

    y_board_strength_flop = torch.tensor(labels["board_strength_flop"], dtype=torch.long, device=device)
    y_board_strength_turn = torch.tensor(labels["board_strength_turn"], dtype=torch.long, device=device)
    y_board_strength_river = torch.tensor(labels["board_strength_river"], dtype=torch.long, device=device)

    kickers_flop_tensor = torch.tensor(labels["hand_values_flop"]).to(device)
    y_flop_kicker_0 = kickers_flop_tensor[:, 1]
    y_flop_kicker_1 = kickers_flop_tensor[:, 2]
    y_flop_kicker_2 = kickers_flop_tensor[:, 3]
    y_flop_kicker_3 = kickers_flop_tensor[:, 4]
    y_flop_kicker_4 = kickers_flop_tensor[:, 5]

    kickers_turn_tensor = torch.tensor(labels["hand_values_turn"]).to(device)
    y_turn_kicker_0 = kickers_turn_tensor[:, 1]
    y_turn_kicker_1 = kickers_turn_tensor[:, 2]
    y_turn_kicker_2 = kickers_turn_tensor[:, 3]
    y_turn_kicker_3 = kickers_turn_tensor[:, 4]
    y_turn_kicker_4 = kickers_turn_tensor[:, 5]

    kickers_river_tensor = torch.tensor(labels["hand_values_river"]).to(device)
    y_river_kicker_0 = kickers_river_tensor[:, 1]
    y_river_kicker_1 = kickers_river_tensor[:, 2]
    y_river_kicker_2 = kickers_river_tensor[:, 3]
    y_river_kicker_3 = kickers_river_tensor[:, 4]
    y_river_kicker_4 = kickers_river_tensor[:, 5]

    kickers_turn_board_tensor = torch.tensor(labels["hand_values_turn_board"]).to(device)
    y_turn_kicker_board_0 = kickers_turn_board_tensor[:, 1]
    y_turn_kicker_board_1 = kickers_turn_board_tensor[:, 2]
    y_turn_kicker_board_2 = kickers_turn_board_tensor[:, 3]
    y_turn_kicker_board_3 = kickers_turn_board_tensor[:, 4]

    kickers_river_board_tensor = torch.tensor(labels["hand_values_river_board"]).to(device)
    y_river_kicker_board_0 = kickers_river_board_tensor[:, 1]
    y_river_kicker_board_1 = kickers_river_board_tensor[:, 2]
    y_river_kicker_board_2 = kickers_river_board_tensor[:, 3]
    y_river_kicker_board_3 = kickers_river_board_tensor[:, 4]
    y_river_kicker_board_4 = kickers_river_board_tensor[:, 5]


    loss_outcome_preflop = F.kl_div(log_probs_preflop, y_outcome_preflop, reduction="batchmean")
    loss_type_preflop = F.cross_entropy(log_probs_type_preflop, y_type_preflop)


    loss_type_flop = F.cross_entropy(log_probs_type_flop, y_type_flop)
    loss_type_turn = F.cross_entropy(log_probs_type_turn, y_type_turn)
    loss_type_river = F.cross_entropy(log_probs_type_river, y_type_river)

    loss_straight_draw_flop = F.cross_entropy(log_probs_straight_draw_flop, y_straight_draw_flop)
    loss_flush_draw_flop = F.cross_entropy(log_probs_flush_draw_flop, y_flush_draw_flop)
    loss_straight_draw_turn = F.cross_entropy(log_probs_straight_draw_turn, y_straight_draw_turn)
    loss_flush_draw_turn = F.cross_entropy(log_probs_flush_draw_turn, y_flush_draw_turn)

    loss_board_strength_flop = F.cross_entropy(log_probs_board_strength_flop, y_board_strength_flop)
    loss_board_strength_turn = F.cross_entropy(log_probs_board_strength_turn, y_board_strength_turn)
    loss_board_strength_river = F.cross_entropy(log_probs_board_strength_river, y_board_strength_river)

    loss_outcome_flop  = F.kl_div(log_probs_flop, y_outcome_flop, reduction="batchmean")
    loss_outcome_turn =  F.kl_div(log_probs_turn, y_outcome_turn, reduction="batchmean")
    loss_outcome_river = F.kl_div(log_probs_river, y_outcome_river, reduction="batchmean")

    loss_flop_kicker_0 = F.cross_entropy(log_probs_flop_kicker_0, y_flop_kicker_0)
    mask_flop_kicker_1 = y_flop_kicker_1 != -2
    loss_flop_kicker_1 = F.cross_entropy(log_probs_flop_kicker_1[mask_flop_kicker_1], y_flop_kicker_1[mask_flop_kicker_1])
    mask_flop_kicker_2 = y_flop_kicker_2 != -2
    loss_flop_kicker_2 = F.cross_entropy(log_probs_flop_kicker_2[mask_flop_kicker_2], y_flop_kicker_2[mask_flop_kicker_2])
    mask_flop_kicker_3 = y_flop_kicker_3 != -2
    loss_flop_kicker_3 = F.cross_entropy(log_probs_flop_kicker_3[mask_flop_kicker_3], y_flop_kicker_3[mask_flop_kicker_3])
    mask_flop_kicker_4 = y_flop_kicker_4 != -2
    loss_flop_kicker_4 = F.cross_entropy(log_probs_flop_kicker_4[mask_flop_kicker_4], y_flop_kicker_4[mask_flop_kicker_4])

    loss_turn_kicker_0 = F.cross_entropy(log_probs_turn_kicker_0, y_turn_kicker_0)
    mask_turn_kicker_1 = y_turn_kicker_1 != -2
    loss_turn_kicker_1 = F.cross_entropy(log_probs_turn_kicker_1[mask_turn_kicker_1], y_turn_kicker_1[mask_turn_kicker_1])
    mask_turn_kicker_2 = y_turn_kicker_2 != -2
    loss_turn_kicker_2 = F.cross_entropy(log_probs_turn_kicker_2[mask_turn_kicker_2], y_turn_kicker_2[mask_turn_kicker_2])
    mask_turn_kicker_3 = y_turn_kicker_3 != -2
    loss_turn_kicker_3 = F.cross_entropy(log_probs_turn_kicker_3[mask_turn_kicker_3], y_turn_kicker_3[mask_turn_kicker_3])
    mask_turn_kicker_4 = y_turn_kicker_4 != -2
    loss_turn_kicker_4 = F.cross_entropy(log_probs_turn_kicker_4[mask_turn_kicker_4], y_turn_kicker_4[mask_turn_kicker_4])

    loss_river_kicker_0 = F.cross_entropy(log_probs_river_kicker_0, y_river_kicker_0)
    mask_river_kicker_1 = y_river_kicker_1 != -2
    loss_river_kicker_1 = F.cross_entropy(log_probs_river_kicker_1[mask_river_kicker_1], y_river_kicker_1[mask_river_kicker_1])
    mask_river_kicker_2 = y_river_kicker_2 != -2
    loss_river_kicker_2 = F.cross_entropy(log_probs_river_kicker_2[mask_river_kicker_2], y_river_kicker_2[mask_river_kicker_2])
    mask_river_kicker_3 = y_river_kicker_3 != -2
    loss_river_kicker_3 = F.cross_entropy(log_probs_river_kicker_3[mask_river_kicker_3], y_river_kicker_3[mask_river_kicker_3])
    mask_river_kicker_4 = y_river_kicker_4 != -2
    loss_river_kicker_4 = F.cross_entropy(log_probs_river_kicker_4[mask_river_kicker_4], y_river_kicker_4[mask_river_kicker_4])


    loss_turn_kicker_board_0 = F.cross_entropy(log_probs_turn_board_kicker_0, y_turn_kicker_board_0)
    mask_turn_kicker_board_1 = y_turn_kicker_board_1 != -2
    loss_turn_kicker_board_1 = F.cross_entropy(
        log_probs_turn_board_kicker_1[mask_turn_kicker_board_1], y_turn_kicker_board_1[mask_turn_kicker_board_1])
    mask_turn_kicker_board_2 = y_turn_kicker_board_2 != -2
    loss_turn_kicker_board_2 = F.cross_entropy(
        log_probs_turn_board_kicker_2[mask_turn_kicker_board_2], y_turn_kicker_board_2[mask_turn_kicker_board_2])
    mask_turn_kicker_board_3 = y_turn_kicker_board_3 != -2
    loss_turn_kicker_board_3 = F.cross_entropy(
        log_probs_turn_board_kicker_3[mask_turn_kicker_board_3], y_turn_kicker_board_3[mask_turn_kicker_board_3])


    loss_river_kicker_board_0 = F.cross_entropy(log_probs_river_board_kicker_0, y_river_kicker_board_0)
    mask_river_kicker_board_1 = y_river_kicker_board_1 != -2
    loss_river_kicker_board_1 = F.cross_entropy(
        log_probs_river_board_kicker_1[mask_river_kicker_board_1], y_river_kicker_board_1[mask_river_kicker_board_1])
    mask_river_kicker_board_2 = y_river_kicker_board_2 != -2
    loss_river_kicker_board_2 = F.cross_entropy(
        log_probs_river_board_kicker_2[mask_river_kicker_board_2], y_river_kicker_board_2[mask_river_kicker_board_2])
    mask_river_kicker_board_3 = y_river_kicker_board_3 != -2
    loss_river_kicker_board_3 = F.cross_entropy(
        log_probs_river_board_kicker_3[mask_river_kicker_board_3], y_river_kicker_board_3[mask_river_kicker_board_3])
    mask_river_kicker_board_4 = y_river_kicker_board_4 != -2
    loss_river_kicker_board_4 = F.cross_entropy(
        log_probs_river_board_kicker_4[mask_river_kicker_board_4], y_river_kicker_board_4[mask_river_kicker_board_4])

    return {
        "outcome_preflop": loss_outcome_preflop,
        "preflop_type": loss_type_preflop,
        "outcome_flop": loss_outcome_flop,
        "type_flop": loss_type_flop,
        "straight_draw_flop": loss_straight_draw_flop,
        "flush_draw_flop": loss_flush_draw_flop,
        "outcome_turn": loss_outcome_turn,
        "type_turn": loss_type_turn,  # This key appears only once now
        "straight_draw_turn": loss_straight_draw_turn,
        "flush_draw_turn": loss_flush_draw_turn,
        "outcome_river": loss_outcome_river,
        "type_river": loss_type_river,
        "board_strength_flop": loss_board_strength_flop,
        "board_strength_turn": loss_board_strength_turn,
        "board_strength_river": loss_board_strength_river,

        "flop_kicker_0": loss_flop_kicker_0,
        "flop_kicker_1": loss_flop_kicker_1,
        "flop_kicker_2": loss_flop_kicker_2,
        "flop_kicker_3": loss_flop_kicker_3,
        "flop_kicker_4": loss_flop_kicker_4,

        # Turn Losses
        "turn_kicker_0": loss_turn_kicker_0,
        "turn_kicker_1": loss_turn_kicker_1,
        "turn_kicker_2": loss_turn_kicker_2,
        "turn_kicker_3": loss_turn_kicker_3,
        "turn_kicker_4": loss_turn_kicker_4,

        # River Losses
        "river_kicker_0": loss_river_kicker_0,
        "river_kicker_1": loss_river_kicker_1,
        "river_kicker_2": loss_river_kicker_2,
        "river_kicker_3": loss_river_kicker_3,
        "river_kicker_4": loss_river_kicker_4,

        # Kicker Board Losses
        "turn_kicker_board_0": loss_turn_kicker_board_0,
        "turn_kicker_board_1": loss_turn_kicker_board_1,
        "turn_kicker_board_2": loss_turn_kicker_board_2,
        "turn_kicker_board_3": loss_turn_kicker_board_3,


        "river_kicker_board_0": loss_river_kicker_board_0,
        "river_kicker_board_1": loss_river_kicker_board_1,
        "river_kicker_board_2": loss_river_kicker_board_2,
        "river_kicker_board_3": loss_river_kicker_board_3,
        "river_kicker_board_4": loss_river_kicker_board_4,

    }



def consumer(queue, model, optimizer, save_interval, save_path, tasks, untrained_model):
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler
    model.to(device)

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    model.train()  # Set model to training mode
    total_samples = 0  # Total number of samples processed
    samples_since_last_save = 0  # Samples processed since the last checkpoint

    ema_alpha = 2/(5_000_000/BATCH_SIZE)  # Smoothing factor (adjust as needed: 0 < alpha <= 1)
    ema_losses = {task: 0 for task in tasks}  # Initialize EMA to 0
    t = 0  # Weight for bias correction
    loss_weights = {}
    weights_initialized = False
    with tqdm(desc="Training Progress", total=None, dynamic_ncols=True) as pbar:
        while True:
            try:
                features, labels = queue.get()  # Blocking call to fetch data from the queue
                features = torch.tensor(features, dtype=torch.long, device=device)
                preflop = features[:, :2]
                flop = features[:, 2:5]
                turn = features[:, 5:6]
                river = features[:, 6:7]


                # Forward and backward passes with mixed precision
                with torch.cuda.amp.autocast():
                    if not weights_initialized:
                        untrained_model.to(device)
                        with torch.no_grad():
                            _, results = untrained_model(preflop, flop, turn, river)
                            losses = calc_losses(results, labels)


                            for task in losses:
                                loss_weights[task] = 1/losses[task].item()
                        weights_initialized = True
                        del(untrained_model)
                        continue

                    kickers_flop_tensor = torch.tensor(labels["hand_values_flop"]).to(device)
                    kickers_turn_tensor = torch.tensor(labels["hand_values_turn"]).to(device)
                    kickers_river_tensor = torch.tensor(labels["hand_values_river"]).to(device)

                    kickers_flop = tuple([kickers_flop_tensor[:, i] for i in range(5)])
                    kickers_turn = tuple([kickers_turn_tensor[:, i] for i in range(5)])
                    kickers_river = tuple([kickers_river_tensor[:, i] for i in range(5)])

                    kickers_turn_board_tensor = torch.tensor(labels["hand_values_turn_board"]).to(device)
                    kickers_river_board_tensor = torch.tensor(labels["hand_values_river_board"]).to(device)

                    kickers_turn_board = tuple([kickers_turn_board_tensor[:, i] for i in range(4)])
                    kickers_river_board = tuple([kickers_river_board_tensor[:, i] for i in range(5)])

                    _, results = model(
                        preflop,
                        flop,
                        turn,
                        river,
                        *kickers_flop,
                        *kickers_turn,
                        *kickers_river,
                        *kickers_turn_board,
                        *kickers_river_board,
                    )

                    losses = calc_losses(results, labels)
                    optimizer.zero_grad()
                    total_loss = 0
                    for task in losses:
                        if task in tasks:
                            total_loss += losses[task] * loss_weights[task]
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # Update metrics
                batch_size_actual = features.size(0)  # Account for smaller last batch
                total_samples += batch_size_actual
                samples_since_last_save += batch_size_actual
                # Update EMA loss
                # Update EMA loss with bias correction
                t += 1

                correction_factor = 1 - (1 - ema_alpha) ** t
                for task, ema_loss in ema_losses.items():
                    ema_losses[task] = ema_alpha * losses[task].item() + (1 - ema_alpha) * ema_loss

                # Update progress bar
                pbar.update(batch_size_actual)
                pbar.set_postfix({
                    "PR": f"{ema_losses['outcome_river'] / correction_factor:.5f}",
                    "TypeR": f"{ema_losses['type_river'] / correction_factor:.5f}",
                    # "PP": f"{ema_losses['outcome_preflop'] / correction_factor:.8f}",
                    #"TypeP": f"{ema_losses['preflop_type'] / correction_factor:.8f}",
                    "PF": f"{ema_losses['outcome_flop'] / correction_factor:.5f}",
                    "TypeF": f"{ema_losses['type_flop'] / correction_factor:.5f}",
                    "SDF": f"{ema_losses['straight_draw_flop'] / correction_factor:.5f}",
                    "FDF": f"{ema_losses['flush_draw_flop'] / correction_factor:.5f}",
                    "PT": f"{ema_losses['outcome_turn'] / correction_factor:.5f}",
                    "TypeT": f"{ema_losses['type_turn'] / correction_factor:.5f}",
                    "SDT": f"{ema_losses['straight_draw_turn'] / correction_factor:.5f}",
                    "FDT": f"{ema_losses['flush_draw_turn'] / correction_factor:.5f}",
                    "boardF": f"{ema_losses['board_strength_flop'] / correction_factor:.5f}",
                    "boardT": f"{ema_losses['board_strength_turn'] / correction_factor:.5f}",
                    "boardR": f"{ema_losses['board_strength_river'] / correction_factor:.5f}",

                    #"FK0": f"{ema_losses['flop_kicker_0'] / correction_factor:.3f}",
                    #"FK1": f"{ema_losses['flop_kicker_1'] / correction_factor:.3f}",
                    #"FK2": f"{ema_losses['flop_kicker_2'] / correction_factor:.3f}",
                    #"FK3": f"{ema_losses['flop_kicker_3'] / correction_factor:.3f}",
                    #"FK4": f"{ema_losses['flop_kicker_4'] / correction_factor:.3f}",

                    # Turn Losses
                    #"TK0": f"{ema_losses['turn_kicker_0'] / correction_factor:.3f}",
                    #"TK1": f"{ema_losses['turn_kicker_1'] / correction_factor:.3f}",
                    #"TK2": f"{ema_losses['turn_kicker_2'] / correction_factor:.3f}",
                    #"TK3": f"{ema_losses['turn_kicker_3'] / correction_factor:.3f}",
                    #"TK4": f"{ema_losses['turn_kicker_4'] / correction_factor:.3f}",

                    # River Losses
                    "RK0": f"{ema_losses['river_kicker_0'] / correction_factor:.3f}",
                    #"RK1": f"{ema_losses['river_kicker_1'] / correction_factor:.3f}",
                    #"RK2": f"{ema_losses['river_kicker_2'] / correction_factor:.3f}",
                    #"RK3": f"{ema_losses['river_kicker_3'] / correction_factor:.3f}",
                    #"RK4": f"{ema_losses['river_kicker_4'] / correction_factor:.3f}",

                    # Board Losses
                    "BTK0": f"{ema_losses['turn_kicker_board_0'] / correction_factor:.3f}",
                    "BTK1": f"{ema_losses['turn_kicker_board_1'] / correction_factor:.3f}",
                    "BTK2": f"{ema_losses['turn_kicker_board_2'] / correction_factor:.3f}",
                    "BTK3": f"{ema_losses['turn_kicker_board_3'] / correction_factor:.3f}",

                    # River Losses
                    "BRK0": f"{ema_losses['river_kicker_board_0'] / correction_factor:.3f}",
                    "BRK1": f"{ema_losses['river_kicker_board_1'] / correction_factor:.3f}",
                    "BRK2": f"{ema_losses['river_kicker_board_2'] / correction_factor:.3f}",
                    "BRK3": f"{ema_losses['river_kicker_board_3'] / correction_factor:.3f}",
                    "BRK4": f"{ema_losses['river_kicker_board_4'] / correction_factor:.3f}",

                })

                # Save model checkpoint periodically
                if samples_since_last_save >= save_interval:
                    save_model(model, optimizer, total_samples, save_path)
                    samples_since_last_save = 0  # Reset save counter

            except Exception as e:
                print(f"Error in consumer: {e}")
                break



def save_model(model, optimizer, total_samples, path="model_checkpoint.pth"):
    torch.save({
        "total_samples": total_samples,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(path, model, optimizer=None, lr=None, strict=True):
    """
    Load a checkpoint into the model and optimizer.
    """

    try:
        checkpoint = torch.load(path) # Load the checkpoint file

        model_state_dict = model.state_dict()

        # Filter out mismatched layers
        filtered_state_dict = {k: v for k, v in checkpoint["model_state_dict"].items() if
                               k in model_state_dict and v.shape == model_state_dict[k].shape}

        # Load the filtered parameters
        model.load_state_dict(filtered_state_dict, strict=False)

        if optimizer and strict:  # If an optimizer is provided, load its state

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        total_samples = checkpoint.get("total_samples", 0)  # Restore training progress
        print(f"Checkpoint loaded. Resumed from {total_samples} samples.")
        return total_samples

    except FileNotFoundError:
        print("No checkpoint found. Starting training from scratch.")
        return 0

NUM_PRODUCERS = 7  # Adjust this to the number of cores you want for producers
BUFFER_SIZE = 300_000  # Buffer size for the multiprocessing queue
BATCH_SIZE = 256  # Batch size for training
SAVE_INTERVAL = 1_000_000  # Save model checkpoint every 1M samples
MODEL_SAVE_PATH = "model_checkpoint.pth"  # Path to save model checkpoints
embedding_dim = 4  # Size of individual card embeddings
# Example usage
feature_dim = 256  # Size of output feature vectors
deep_layer_dims = (512, 2048, 2048, 2048)
intermediary_dim = 16
load = True
strict = True
if __name__ == "__main__":
    tasks = [
        "outcome_preflop",
        "outcome_flop",
        "preflop_type",
        "type_flop",
        "straight_draw_flop",
        "flush_draw_flop",
        "outcome_turn",
        "type_turn",
        "straight_draw_turn",
        "flush_draw_turn",
        "outcome_river",
        "type_river",
        "board_strength_flop",
        "board_strength_turn",
        "board_strength_river",
        "flop_kicker_0",
        "flop_kicker_1",
        "flop_kicker_2",
        "flop_kicker_3",
        "flop_kicker_4",
        "turn_kicker_0",
        "turn_kicker_1",
        "turn_kicker_2",
        "turn_kicker_3",
        "turn_kicker_4",
        "river_kicker_0",
        "river_kicker_1",
        "river_kicker_2",
        "river_kicker_3",
        "river_kicker_4",

        # Kicker Board Losses
        "turn_kicker_board_0",
        "turn_kicker_board_1",
        "turn_kicker_board_2",
        "turn_kicker_board_3",

        "river_kicker_board_0",
        "river_kicker_board_1",
        "river_kicker_board_2",
        "river_kicker_board_3",
        "river_kicker_board_4",
    ]
    # Use 'spawn' to support CUDA with multiprocessing
    set_start_method("spawn", force=True)

    # Initialize model, loss function, and optimizer
    model = PokerHandEmbedding(embedding_dim, feature_dim, deep_layer_dims, intermediary_dim)
    untrained_model = PokerHandEmbedding(embedding_dim, feature_dim, deep_layer_dims, intermediary_dim)
    #optimizers = {task: torch.optim.AdamW(model.parameters(), lr=0.0001) for task in tasks}
    optimizer =  torch.optim.AdamW(model.parameters(), lr=0.0001)

    # Load checkpoint (optional)
    if load:
        total_samples = load_checkpoint(MODEL_SAVE_PATH, model, optimizer, strict=strict, lr=0.00001)

    # Initialize multiprocessing queue and event
    remainder = BUFFER_SIZE % BATCH_SIZE
    true_buffer_size = BUFFER_SIZE + (BATCH_SIZE - remainder) if remainder != 0 else BUFFER_SIZE
    queue = Queue(maxsize=true_buffer_size)


    producers = []

    for i in range(NUM_PRODUCERS):
        p = Process(target=producer, args=(queue, BATCH_SIZE))
        p.daemon = True
        p.start()
        producers.append(p)



    # Define and start the consumer process
    consumer_process = Process(
        target=consumer,
        args=(queue, model, optimizer, SAVE_INTERVAL, MODEL_SAVE_PATH, tasks, untrained_model),
    )
    consumer_process.daemon = True
    consumer_process.start()
    print("Consumer process started.")

    # Manage processes and handle termination
    try:
        for p in producers:
            p.join()
        consumer_process.join()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Terminating processes...")
        for p in producers:
            p.terminate()
            p.join()
        consumer_process.terminate()
        consumer_process.join()
        print("Processes terminated. Exiting.")

