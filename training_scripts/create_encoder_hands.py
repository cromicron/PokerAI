import sys
import os
from pathlib import Path
# Add the 'src' directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from tqdm import tqdm

import random
from multiprocessing import Process, Queue, set_start_method
from collections import Counter
from itertools import combinations
import numpy as np
from pokerAI.lookup.HandComparatorLookup import  strength, strength_array
from pokerAI.lookup.lookup_probs_flop import get_probs_flop
from PokerGame.HandComperator import strength as strength_old
import pickle



# Get the absolute path to the src directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))

# Construct the correct path to the pickle file
pkl_path = os.path.join(BASE_DIR, "pokerAI/lookup/preflop_probs.pkl")


with open(pkl_path, "rb") as f:
    LOOKUP_PREFLOP = pickle.load(f)


deck = [(rank, suit) for rank in range(2, 15) for suit in range(4)]
card_to_int = {
    card: i for i, card in enumerate(deck)
}

in_to_card = {
    i: card for i, card in enumerate(deck)
}
COMBO_INDICES_RIVER = np.array(list(combinations(range(45), 2)))
N_SIM_TURN = 10_000
COMBO_INDICES_TURN = np.stack([np.random.choice(range(46), 3, replace=False) for _ in range(N_SIM_TURN)])
MENAINGLESS_CARDS = [(-7, -11), (-99, -9), (-321, -77)]




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



def consumer(queue, batches_per_file=100, n_files=10_000):
    feature_dump = []
    label_dump = []
    data_idx = 0
    for _ in tqdm(range(n_files)):
        try:
            features, labels = queue.get()  # Blocking call to fetch data from the queue
            feature_dump.append(features)
            label_dump.append(labels)
            if len(feature_dump) == batches_per_file:
                feature_data = np.vstack(feature_dump)
                file_path_features = Path(PATH_DATA) / f"features_{data_idx:04d}.npy"
                np.save(file_path_features, feature_data)
                label_data = {key: np.concatenate([d[key] for d in label_dump]) for key in label_dump[0]}
                file_path_labels = Path(PATH_DATA) / f"labels_{data_idx:04d}.pkl"
                with file_path_labels.open("wb") as f:
                    pickle.dump(label_data, f)
                data_idx += 1
                print("saved data " + str(data_idx))
                feature_dump = []
                label_dump = []


        except Exception as e:
            print(f"Error in consumer: {e}")
            break



NUM_PRODUCERS = 30  # Adjust this to the number of cores you want for producers
BUFFER_SIZE = 1_000_000  # Buffer size for the multiprocessing queue
BATCH_SIZE = 1024  # Batch size for training
PATH_DATA = "../data"
n_files = 10_000


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
        args=(queue,100),
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

