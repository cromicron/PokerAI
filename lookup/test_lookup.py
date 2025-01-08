from PokerGame.HandComperator import strength as old_strength
from lookup.HandComparatorLookup import strength, encode_hand
from PokerGame.NLHoldem import Deck
import pickle
import numpy as np
import random
from tqdm import tqdm

with open("flush_lookup.pkl", "rb") as f:
    suit_dict = pickle.load(f)
with open("lookup.pkl", "rb") as f:
    strength_dict = pickle.load(f)
deck = Deck()
for i in tqdm(range(10000000)):
    random.shuffle(deck)
    hero = [(card.value, card.suit) for card in deck[:7]]
    villain = [(card.value, card.suit) for card in deck[7:14]]

    hero.sort(reverse=True)

    villain.sort(reverse=True)
    suits_hero = tuple([card[1] for card in hero])
    suits_villain = tuple([card[1] for card in villain])

    ranks_hero = tuple([card[0] for card in hero])
    ranks_villain = tuple([card[0] for card in villain])

    encoded_hand_hero = encode_hand(ranks_hero, suit_dict[suits_hero])
    encoded_hand_villain = encode_hand(ranks_villain, suit_dict[suits_villain])

    encoded_hands = [encoded_hand_hero, encoded_hand_villain]

    try:
        strengths_lookup = [strength_dict[hand] for hand in encoded_hands]
    except:
        print(hero, villain)

    if strengths_lookup[0] > strengths_lookup[1]:
        result_lookup = 1
    elif strengths_lookup[0] == strengths_lookup[1]:
        result_lookup = 0
    else:
        result_lookup = 2

    old_strengths = [old_strength(hand) for hand in [hero, villain]]
    if old_strengths[0] > old_strengths[1]:
        result_old = 1
    elif old_strengths[0] == old_strengths[1]:
        result_old = 0
    else:
        result_old = 2
    if result_old != result_lookup:
        print(hero)
        print(villain)








