import unittest
from PokerGame.NLHoldem import Game
from Agents.Regret import Agent

class TestPoker(unittest.TestCase):
    def test_create_state_init(self):
        stacks = [100, 25]
        game = Game(n_players=2, stacks=stacks, hand_history=True)
        game.new_hand(first_hand=True)
        agent_0 = Agent()
        agent_1 = Agent()
        map_agent = {game.players[0]: agent_0, game.players[1]: agent_1}
        hand_0 = game.players[0].holecards
        hand_1 = game.players[1].holecards
        cards_0 = agent_0.one_hot_encode_hand(*hand_0)
        cards_1 = agent_1.one_hot_encode_hand(*hand_1)
        map_cards = {agent_0: cards_0, agent_1: cards_1}
        next = game.next[-1]
        agent_to_act = map_agent[next]
        cards = map_cards[agent_to_act]
        hand_history = game.hand_history.hand_history
        position = game.positions.index(next)

        state = agent_to_act.create_state(cards, position, hand_history)
        holecard_0, holecard_1 = game.next[-1].holecards
        val_0, suit_0 = holecard_0.value, holecard_0.suit
        val_1, suit_1 = holecard_1.value, holecard_1.suit
        index_card_0 = (val_0 - 2)*4 +suit_0
        index_card_1 = (val_1 - 2) * 4 + suit_1
        self.assertEqual(state[index_card_0],1)
        self.assertEqual(state[index_card_1], 1)
        self.assertEqual(state[:52].sum(), 2)
        suited = int(suit_0 == suit_1)
        self.assertEqual(suited, state[52])
        self.assertEqual(state[211], 100)
        self.assertEqual(state[212], 25)
        self.assertEqual(state[213], 0)
        self.assertEqual(state[214], 99)
        self.assertEqual(state[215], 23)
        self.assertEqual(state[216], 1)
        self.assertEqual(state[217], 2)
        self.assertEqual(state[218], 3)
        game.implement_action(next, 1)

        next = game.next[-1]
        agent_to_act = map_agent[next]
        cards = map_cards[agent_to_act]
        hand_history = game.hand_history.hand_history
        position = game.positions.index(next)
        state = agent_to_act.create_state(cards, position, hand_history)
        holecard_0, holecard_1 = game.next[-1].holecards
        val_0, suit_0 = holecard_0.value, holecard_0.suit
        val_1, suit_1 = holecard_1.value, holecard_1.suit
        index_card_0 = (val_0 - 2)*4 +suit_0
        index_card_1 = (val_1 - 2) * 4 + suit_1
        self.assertEqual(state[index_card_0],1)
        self.assertEqual(state[index_card_1], 1)
        self.assertEqual(state[:52].sum(), 2)
        suited = int(suit_0 == suit_1)
        self.assertEqual(suited, state[52])
        self.assertEqual(state[210], 1)
        self.assertEqual(state[211], 25)
        self.assertEqual(state[212], 100)
        self.assertEqual(state[214], 23)
        self.assertEqual(state[215], 98)
        self.assertEqual(state[216], 2)
        self.assertEqual(state[217], 2)
        self.assertEqual(state[218], 4)
        self.assertEqual(state[219], 0)
        self.assertEqual(state[220], 0)
        self.assertEqual(state[221], 1)

        game = Game(n_players=2, stacks=stacks, hand_history=True)
        game.new_hand(first_hand=True)
        agent_0 = Agent()
        agent_1 = Agent()
        map_agent = {game.players[0]: agent_0, game.players[1]: agent_1}
        hand_0 = game.players[0].holecards
        hand_1 = game.players[1].holecards
        cards_0 = agent_0.one_hot_encode_hand(*hand_0)
        cards_1 = agent_1.one_hot_encode_hand(*hand_1)
        map_cards = {agent_0: cards_0, agent_1: cards_1}
        next = game.next[-1]
        game.implement_action(next, 2, 5)

        next = game.next[-1]
        agent_to_act = map_agent[next]
        cards = map_cards[agent_to_act]
        hand_history = game.hand_history.hand_history
        position = game.positions.index(next)
        state = agent_to_act.create_state(cards, position, hand_history)
        holecard_0, holecard_1 = game.next[-1].holecards
        val_0, suit_0 = holecard_0.value, holecard_0.suit
        val_1, suit_1 = holecard_1.value, holecard_1.suit
        index_card_0 = (val_0 - 2)*4 +suit_0
        index_card_1 = (val_1 - 2) * 4 + suit_1
        self.assertEqual(state[index_card_0],1)
        self.assertEqual(state[index_card_1], 1)
        self.assertEqual(state[:52].sum(), 2)
        suited = int(suit_0 == suit_1)
        self.assertEqual(suited, state[52])
        self.assertEqual(state[210], 1)
        self.assertEqual(state[211], 25)
        self.assertEqual(state[212], 100)
        self.assertEqual(state[214], 23)
        self.assertEqual(state[215], 95)
        self.assertEqual(state[216], 2)
        self.assertEqual(state[217], 5)
        self.assertEqual(state[218], 7)
        self.assertEqual(state[219], 0)
        self.assertEqual(state[220], 0)
        self.assertEqual(state[221], 0)
        self.assertEqual(state[222], 5)

    def test_create_state_complex(self):
        stacks = [100, 25]
        game = Game(n_players=2, stacks=stacks, hand_history=True)
        game.new_hand(first_hand=True)
        agent_0 = Agent()
        agent_1 = Agent()
        map_agent = {game.players[0]: agent_0, game.players[1]: agent_1}
        hand_0 = game.players[0].holecards
        hand_1 = game.players[1].holecards
        cards_0 = agent_0.one_hot_encode_hand(*hand_0)
        cards_1 = agent_1.one_hot_encode_hand(*hand_1)
        map_cards = {agent_0: cards_0, agent_1: cards_1}
        next = game.next[-1]
        agent_to_act = map_agent[next]
        cards = map_cards[agent_to_act]
        hand_history = game.hand_history.hand_history
        position = game.positions.index(next)

        state = agent_to_act.create_state(cards, position, hand_history)
        game.implement_action(next, 1)
        agent_to_act.add_to_episode(state, 1, None)

        next = game.next[-1]
        agent_to_act = map_agent[next]
        cards = map_cards[agent_to_act]
        hand_history = game.hand_history.hand_history
        position = game.positions.index(next)
        state = agent_to_act.create_state(cards, position, hand_history)
        game.implement_action(next, 1)
        agent_to_act.add_to_episode(state, 1, None)

        state = agent_to_act.create_state(cards, position, hand_history, state)
        self.assertEqual(state[213], 1)
        self.assertEqual(state[214], 23)
        self.assertEqual(state[215], 98)
        self.assertEqual(state[216], 0)
        self.assertEqual(state[217], 0)
        self.assertEqual(state[218], 4)
        self.assertEqual(state[219], 1)
        self.assertEqual(state[220], 0)
        self.assertEqual(state[221], 0)
        self.assertEqual(state[222], 0)
        game.implement_action(next, 2, 3)
        agent_to_act.add_to_episode(state, 2, None)

        next = game.next[-1]
        agent_to_act = map_agent[next]
        cards = map_cards[agent_to_act]
        hand_history = game.hand_history.hand_history
        position = game.positions.index(next)
        last_state = agent_to_act.episode["inputs"][-1]
        state = agent_to_act.create_state(cards, position, hand_history, last_state)
        self.assertEqual(state[213], 1)
        self.assertEqual(state[214], 98)
        self.assertEqual(state[215], 20)
        self.assertEqual(state[216], 0)
        self.assertEqual(state[217], 3)
        self.assertEqual(state[218], 7)
        self.assertEqual(state[219], 1)
        self.assertEqual(state[220], 0)
        self.assertEqual(state[221], 1)
        self.assertEqual(state[222], 0)
        self.assertEqual(state[223], 0)
        self.assertEqual(state[224], 3)
        game.implement_action(next, 2, 10)
        agent_to_act.add_to_episode(state, 2, None)

        next = game.next[-1]
        agent_to_act = map_agent[next]
        cards = map_cards[agent_to_act]
        hand_history = game.hand_history.hand_history
        position = game.positions.index(next)
        last_state = agent_to_act.episode["inputs"][-1]
        state = agent_to_act.create_state(cards, position, hand_history, last_state)
        self.assertEqual(state[213], 1)
        self.assertEqual(state[214], 20)
        self.assertEqual(state[215], 88)
        self.assertEqual(state[216], 3)
        self.assertEqual(state[217], 10)
        self.assertEqual(state[218], 17)
        self.assertEqual(state[219], 0)
        self.assertEqual(state[220], 3)
        self.assertEqual(state[221], 0)
        self.assertEqual(state[222], 10)
        self.assertEqual(state[223], 0)
        self.assertEqual(state[224], 0)
        game.implement_action(next, 1)
        agent_to_act.add_to_episode(state, 1, None)

        next = game.next[-1]
        agent_to_act = map_agent[next]
        cards = map_cards[agent_to_act]
        hand_history = game.hand_history.hand_history
        position = game.positions.index(next)
        last_state = agent_to_act.episode["inputs"][-1]
        state = agent_to_act.create_state(cards, position, hand_history, last_state)
        self.assertEqual(state[213], 2)
        self.assertEqual(state[214], 13)
        self.assertEqual(state[215], 88)
        self.assertEqual(state[216], 0)
        self.assertEqual(state[217], 0)
        self.assertEqual(state[218], 24)
        self.assertEqual(state[219], 1)
        self.assertEqual(state[220], 0)
        self.assertEqual(state[221], 0)
        self.assertEqual(state[222], 0)
        self.assertEqual(state[223], 0)
        self.assertEqual(state[224], 0)
