import unittest
from Agent import Agent
import os, sys
from itertools import combinations
import random
p = os.path.abspath('/home/cromi/Documents/Code/Python/PokerAI/PokerGame')
sys.path.insert(1, p)
from NLHoldem import Game, Deck
import copy
import numpy as np

agent = Agent()
input_dims =25
agent.create_action_network(input_dims)
agent.create_bet_network(input_dims)
deck = copy.deepcopy(Deck())
cards = sorted([(card.value, card.suit)for card in deck])
holecards = list(combinations(cards,2))
holecards_dict = dict(zip(holecards,list(range(len(holecards)))))

class TestAgent(unittest.TestCase):
    def test_range_init(self):
        #range hero means it is the range hero has from villains perspective
        game = Game()
        game.new_hand(first_hand=True)
        hero = game.next[-1]
        hole = sorted([(hero.holecards[0].value, hero.holecards[0].suit),(hero.holecards[1].value, hero.holecards[1].suit)])
        agent.initialize_ranges(hole)
        range_hero = agent.range_hero.range
        range_villain = agent.range_villain.range
        self.assertEqual(range_hero[holecards_dict[tuple(hole)]],1/1326)
        self.assertEqual(range_villain[holecards_dict[tuple(hole)]],0)

        sample_cards = random.sample(holecards,1)
        sample_cards = list(sample_cards[0])
        sample_cards.sort()
        while sample_cards == hole:
            sample_cards = random.sample(holecards,1)
            sample_cards = list(sample_cards[0])
            sample_cards.sort()
        self.assertTrue(tuple(sample_cards) in holecards_dict.keys())
        self.assertEqual(range_villain[holecards_dict[tuple(sample_cards)]],1/1324)

    def test_range_update(self):
        game = Game()
        game.new_hand(first_hand=True)
        hero = game.next[-1]
        hole = sorted([(hero.holecards[0].value, hero.holecards[0].suit),(hero.holecards[1].value, hero.holecards[1].suit)])
        agent.initialize_ranges(hole)

        observation = game.get_observation()


        # hero is the first to act. Villain has no info on hero's hand. Therefore we can add blanks into obs
        observation[5:9] = [0,0,0,0]

