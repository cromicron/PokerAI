import unittest
from PokerGame.NLHoldem import Game
from PokerGame.HandComperator import compare_hands
from PokerGame.NLHoldem import SizeError
from PokerGame.NLHoldem import Card


class TestPoker(unittest.TestCase):
    def test_deck_52_cards(self):
        game = Game()
        self.assertTrue(len(game.deck) == 52)

    def test_shuffle(self):
        game = Game()
        cards_before = [(card.value, card.suit) for card in game.deck]
        game.deck.shuffle()
        cards_after = [(card.value, card.suit) for card in game.deck]
        self.assertSetEqual(set(cards_before), set(cards_after))
        self.assertNotEqual(cards_before, cards_after)

    def test_starting_stack_numbers(self):
        self.assertRaises(ArithmeticError, Game, 2,[20,30,20])

    def test_invalid_starting_stack(self):
        self.assertRaises(ValueError, Game, 2, [20, 3.8])

    def test_new_hand(self):
        #more than 2 players
        game = Game(n_players=7)
        game.new_hand()
        self.assertTrue(game.positions[0].holecards == game.deck[0:2])
        self.assertTrue(game.positions[1].holecards == game.deck[2:4])
        self.assertTrue(game.board == [])
        self.assertTrue(game.positions[0].stack == 199)
        self.assertTrue(game.positions[1].stack == 198)
        self.assertTrue(game.pot == 3)
        self.assertTrue(game.next[-1] == game.positions[2])
        #rotating when calling new_hand again:
        expected_positions = [game.positions[1], game.positions[2], game.positions[3], game.positions[4],
                              game.positions[5], game.positions[6], game.positions[0]]
        game.new_hand()
        self.assertEqual(expected_positions,list(game.positions))
        self.assertEqual(game.next[-1], expected_positions[2])
        self.assertEqual(game.next[-2], expected_positions[3])
        self.assertEqual(game.next[-3], expected_positions[4])
        self.assertEqual(game.next[-4], expected_positions[5])
        self.assertEqual(game.next[-5], expected_positions[6])
        self.assertEqual(game.next[-7], expected_positions[1])
        self.assertEqual(game.next[-6], expected_positions[0])

        #two players
        game = Game(n_players=2)
        game.new_hand()
        self.assertTrue(game.positions[0].holecards == game.deck[0:2])
        self.assertTrue(game.positions[1].holecards == game.deck[2:4])
        self.assertTrue(game.board == [])
        self.assertTrue(game.positions[0].stack == 199)
        self.assertTrue(game.positions[1].stack == 198)
        self.assertTrue(game.pot == 3)
        self.assertTrue(game.next[-1] == game.positions[0])
        #rotating when calling new_hand again:
        expected_positions = [game.positions[1], game.positions[0]]
        game.new_hand()
        self.assertEqual(expected_positions,list(game.positions))
        self.assertEqual(game.next[-1], expected_positions[0])
        self.assertEqual(game.next[0], expected_positions[1])


    def test_compare_mult(self):
        #create a couple of boards
        #for straight flush
        ht = Card(10,0)
        h9 = Card(9,0)
        h6 = Card(6,0)
        ct = Card(10,1)
        hj = Card(11,0)
        board = [ht,h9,h6,ct,hj]
        hands = [[Card(12,0), Card(13,0)],[Card(7,0),Card(8,0)],[Card(10,3), Card(10,2)],[Card(11,1), Card(11,3)],
                 [Card(14,0), Card(2,0)],[Card(7,1), Card(8,2)],[Card(14,1), Card(14,2)]]
        for hand in hands:
            hand.extend(board)
        strengths = compare_hands(hands)
        self.assertEqual(int(strengths[0]),8)
        self.assertEqual(int(strengths[1]),8)
        self.assertEqual(int(strengths[2]),7)
        self.assertEqual(int(strengths[3]),6)
        self.assertEqual(int(strengths[4]),5)
        self.assertEqual(int(strengths[5]),4)
        self.assertEqual(int(strengths[6]),2)
        self.assertTrue(strengths[0]>strengths[1])
        self.assertTrue(strengths[1]>strengths[2])
        self.assertTrue(strengths[2]>strengths[3])
        self.assertTrue(strengths[3]>strengths[4])
        self.assertTrue(strengths[4]>strengths[5])
        self.assertTrue(strengths[5]>strengths[6])

    def test_implement_action_incorrect_player(self):
        game = Game()
        game.new_hand()
        self.assertRaises(ValueError, game.implement_action, game.next[0],1)

    def test_get_legal_actions_start(self):
        #sb
        game = Game()
        game.new_hand(first_hand=True)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions, (0,1,2))
        game = Game(2, [4,20])
        game.positions[0].stack=1
        game.new_hand(first_hand=True,reset_to_starting=False)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions, (0,1))
        #bb after call
        game=Game(2, [20,20])
        game.new_hand(first_hand = True)
        game.positions[0].stack -= 1
        game.pot += 1
        game.next.pop()
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(1,2))
        ##bb after raise stack left
        game=Game(2)
        game.new_hand(first_hand=True)
        game.positions[0].stack -= 7
        game.pot += 7
        game.max_bet = 8
        game.next.pop()
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(0,1,2))

        ##bb after raise no stack left
        game=Game(2, [200, 10])
        game.new_hand(first_hand=True)
        game.positions[0].stack -= 21
        game.pot += 21
        game.max_bet = 22
        game.next.pop()
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(0,1))

    def test_legal_actions_post(self):
    #post_flop -first in
        game= Game(4)
        game.new_hand(first_hand=True)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(0,1,2))
        game.implement_action(game.next[-1],1)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(0,1,2))
        game.implement_action(game.next[-1],1)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(0,1,2))
        game.implement_action(game.next[-1],1)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(1,2))
        game.implement_action(game.next[-1],1)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(1,2))
        game.implement_action(game.next[-1],1)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(1,2))
        game.implement_action(game.next[-1],1)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(1,2))

    def test_legal_actions_post_shortstack(self):
        stacks = [20, 30, 40 ,50]
        game = Game(4, stacks=stacks)
        game.new_hand(first_hand=True)
        game.implement_action(game.next[-1], 1)
        game.implement_action(game.next[-1], 2, 10)
        game.implement_action(game.next[-1], 2, 20)
        game.implement_action(game.next[-1], 0)
        game.implement_action(game.next[-1], 0)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions, (0,1))

    def test_legal_actions_HU(self):
        #post_flop -first in
        game= Game(2)
        game.new_hand(first_hand=True)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(0,1,2))
        game.implement_action(game.next[-1],1)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(1,2))
        game.implement_action(game.next[-1],1)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(1,2))
        game.implement_action(game.next[-1],1)
        legal_actions = game.get_legal_actions()
        self.assertEqual(legal_actions,(1,2))
        game.implement_action(game.next[-1],1)

    def test_get_legal_betsize_hu(self):
        game= Game(2)
        game.new_hand(first_hand=True)
        legal_bet = game.get_legal_betsize()
        self.assertEqual(legal_bet, 4)
        game.implement_action(game.next[-1],2,4)
        legal_bet = game.get_legal_betsize()
        self.assertEqual(legal_bet, 6)
        game.implement_action(game.next[-1],2,30)
        legal_bet=game.get_legal_betsize()
        self.assertEqual(legal_bet, 56)
        game.implement_action(game.next[-1],1)
        legal_bet=game.get_legal_betsize()
        self.assertEqual(legal_bet, 2)
        game.implement_action(game.next[-1],2,164)
        legal_bet=game.get_legal_betsize()
        self.assertEqual(legal_bet, 170)

    def test_get_legal_betsize_different_stack(self):
        stacks = [100, 200, 20, 40, 60]
        game = Game(5, stacks)
        game.new_hand(first_hand=True)
        legal_size = game.get_legal_betsize()
        self.assertEqual(legal_size, 4)
        game.implement_action(game.next[-1], 2, 4)
        legal_size = game.get_legal_betsize()
        self.assertEqual(legal_size, 6)
        game.implement_action(game.next[-1], 2, 11)
        added_expected = 7
        self.assertEqual(added_expected, game.added)
        legal_size = game.get_legal_betsize()
        self.assertEqual(legal_size, 18)
        game.implement_action(game.next[-1], 1)
        legal_size = game.get_legal_betsize()
        self.assertEqual(legal_size, 18)



    def test_fold(self):
        #fold first in cutoff
        game = Game(n_players=4)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],0)
        next = game.next[-1]
        self.assertEqual(next, game.positions[3])
        self.assertEqual(len(game.next),3)
        self.assertFalse(game.finished)
        self.assertEqual(game.max_bet,2)

        #fold second in bu
        game.implement_action(game.positions[3],0)
        next = game.next[-1]
        self.assertEqual(next, game.positions[0])
        self.assertEqual(len(game.next),2)
        self.assertFalse(game.finished)
        self.assertEqual(game.max_bet,2)

        #fold third in sb
        game.implement_action(game.positions[0],0)
        self.assertEqual(len(game.next),0)
        self.assertTrue(game.finished)
        self.assertEqual(game.positions[1].stack,201)
        self.assertEqual(game.positions[0].stack,199)


    def test_call_check_trough_HU(self):
        #all players call and bb checks behind
        game = Game(2)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[1],1)

        self.assertEqual(game.street,1)
        #deal flop
        expected_board = game.deck[4:7]
        self.assertEqual(game.board, expected_board)
        self.assertEqual(game.next[-1], game.positions[1])
        self.assertEqual(game.pot, 4)
        self.assertEqual(game.max_bet,0)
        self.assertEqual(game.positions[0].stack, 198)
        self.assertEqual(game.positions[1].stack, 198)
        self.assertEqual(game.positions[0].bet,0)
        self.assertEqual(game.positions[1].bet,0)

        #all check flop
        game.implement_action(game.positions[1],1)
        game.implement_action(game.positions[0],1)
        expected_board = game.deck[4:8]
        self.assertEqual(game.board, expected_board)

        #all check turn
        game.implement_action(game.positions[1],1)
        game.implement_action(game.positions[0],1)
        expected_board = game.deck[4:9]
        self.assertEqual(game.board, expected_board)
        #all check river
        game.implement_action(game.positions[1],1)
        game.implement_action(game.positions[0],1)

        #showdown
        hole_0 = game.positions[0].holecards
        hole_1 = game.positions[1].holecards
        community = game.deck[4:9]
        strengths = compare_hands([hole_0+community, hole_1+community])
        if strengths[0] > strengths[1]:
            self.assertTrue(game.positions[0].stack ==202 and game.positions[1].stack==198)
        elif strengths[1] > strengths[0]:
            self.assertTrue(game.positions[1].stack ==202 and game.positions[0].stack==198)
        else:
            self.assertTrue(game.positions[1].stack ==200 and game.positions[0].stack==200)


    def test_call_check_trough_multi(self):
        #all players call and bb checks behind
        game = Game(4)
        game.new_hand(first_hand=True, random_seat=True)
        game.implement_action(game.positions[2],1)
        game.implement_action(game.positions[3],1)
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[1],1)
        self.assertEqual(game.street,1)
        #deal flop
        expected_board = game.deck[8:11]
        self.assertEqual(game.board, expected_board)
        self.assertEqual(game.next[-1], game.positions[0])
        self.assertEqual(game.pot, 8)
        self.assertEqual(game.max_bet,0)
        self.assertEqual(game.positions[0].stack, 198)
        self.assertEqual(game.positions[1].stack, 198)
        self.assertEqual(game.positions[2].stack, 198)
        self.assertEqual(game.positions[3].stack, 198)
        self.assertEqual(game.positions[0].bet,0)
        self.assertEqual(game.positions[1].bet,0)
        self.assertEqual(game.positions[2].bet,0)
        self.assertEqual(game.positions[3].bet,0)

        #all check flop
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[1],1)
        game.implement_action(game.positions[2],1)
        game.implement_action(game.positions[3],1)
        expected_board = game.deck[8:12]
        self.assertEqual(game.board, expected_board)

        #all check turn
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[1],1)
        game.implement_action(game.positions[2],1)
        game.implement_action(game.positions[3],1)
        expected_board = game.deck[8:13]
        self.assertEqual(game.board, expected_board)
        #all check river
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[1],1)
        game.implement_action(game.positions[2],1)
        game.implement_action(game.positions[3],1)

        holecards = [game.positions[i].holecards for i in range(game.n_players)]
        community = game.deck[8:13]

        hands = [holecards[i] + community for i in range(game.n_players)]
        strengths = compare_hands(hands)
        #find best hand
        winners = []
        best = max(strengths)
        winners.append(strengths.index(best))
        counter = winners[-1]+1
        while counter < len(strengths):
            if strengths[counter] == best:
                winners.append(counter)
            counter += 1
        expected_stack_winners = 198+(8/len(winners))
        expected_stack_loosers = 198

        for i in range(game.n_players):
            if i in winners:
                self.assertEqual(game.positions[i].stack,expected_stack_winners)
            else:
                self.assertEqual(game.positions[i].stack,expected_stack_loosers)

    def test_error_illegal_raise(self):
        game = Game(3)
        game.new_hand(first_hand=True)
        self.assertRaises(TypeError,game.implement_action, *(game.positions[2],2)) #when player raises, he must also
        #specify betsize
        #if player doesn't have enough chips to make the bet
        game = Game(3)
        game.new_hand(first_hand=True)
        self.assertRaises(SizeError,game.implement_action, *(game.positions[2],2,202))
        #however he can go all in with all of his chips
        game = Game(3)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],0)
        game.implement_action(game.positions[0],2,200) #no error should be raised


        game = Game(3)
        game.new_hand(first_hand=True)
        self.assertRaises(SizeError,game.implement_action, *(game.positions[2],2,200.2))

        # specify size
        game = Game(3)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],2,8)
        self.assertRaises(SizeError,game.implement_action, *(game.positions[0],2,200.2))

        game = Game(3)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],2,6)
        self.assertRaises(SizeError,game.implement_action, *(game.positions[0],2,5.8))

        game = Game(3)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],2,6)
        game.implement_action(game.positions[0],2,10)
        self.assertRaises(SizeError,game.implement_action, *(game.positions[1],2,12.8))

        game = Game(3)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],2,6)
        game.implement_action(game.positions[0],2,10)
        game.implement_action(game.positions[1],2,14)
        game.implement_action(game.positions[2],1)
        game.implement_action(game.positions[0],1)
        self.assertRaises(SizeError,game.implement_action, *(game.positions[0],2,1.8))

        game = Game(3)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],2,6)
        game.implement_action(game.positions[0],2,10)
        game.implement_action(game.positions[1],2,14)
        game.implement_action(game.positions[2],1)
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[0],2, 30)
        self.assertRaises(SizeError,game.implement_action, *(game.positions[1],2,58))

    def test_correct_betsize_after_flop(self):
        game = Game(3)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],2,6)
        game.implement_action(game.positions[0],2,10)
        game.implement_action(game.positions[1],2,14)
        game.implement_action(game.positions[2],1)
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[0],2, 2)
        game.implement_action(game.positions[1],0)

    def test_correct_deque(self):
        game = Game(4)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],2,6)
        self.assertEqual(list(game.next),[game.positions[1],game.positions[0],game.positions[3]])
        game.implement_action(game.positions[3],2, 18)
        self.assertEqual(list(game.next),[game.positions[2], game.positions[1],game.positions[0]])
        game.implement_action(game.positions[0],1)
        self.assertEqual(list(game.next),[game.positions[2], game.positions[1]])
        game.implement_action(game.positions[1],2,40)
        self.assertEqual(list(game.next),[game.positions[0],game.positions[3],game.positions[2]])
        game.implement_action(game.positions[2],0)
        game.implement_action(game.positions[3],1)
        game.implement_action(game.positions[0],1)



    def test_raise_correct_stack_and_pot_sizes(self):
        game = Game(4)
        game.new_hand(first_hand=True)
        game.implement_action(game.positions[2],2,6)
        game.implement_action(game.positions[3],2, 18)
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[1],2,40)
        game.implement_action(game.positions[2],0)
        game.implement_action(game.positions[3],1)
        game.implement_action(game.positions[0],1)
        stack_sb = game.positions[0].stack
        stack_bb = game.positions[1].stack
        stack_co = game.positions[2].stack
        stack_bu = game.positions[3].stack
        self.assertEqual(stack_sb,160)
        self.assertEqual(stack_bb, 160)
        self.assertEqual(stack_co,194)
        self.assertEqual(stack_bu,160)

    def test_correct_player_after_allin(self):
        #test if a player who goes all in or calls all-in is removed from being next.
        game = Game(4,[200,160,120,20])
        game.new_hand(first_hand=True)
        game.implement_action(game.next[-1],2,120)
        game.implement_action(game.next[-1],1)
        game.implement_action(game.next[-1],1)
        game.implement_action(game.next[-1],2,160)
        self.assertFalse(game.positions[2] in game.next)
        self.assertFalse(game.positions[3] in game.next)


    def test_showdown_when_all_allin(self):
        game = Game(4,[200,160,120,20])
        game.new_hand(first_hand=True)
        game.implement_action(game.next[-1],2,120)
        game.implement_action(game.next[-1],1)
        game.implement_action(game.next[-1],1)
        game.implement_action(game.next[-1],2,160)
        game.implement_action(game.next[-1],1)
        self.assertTrue(game.finished)

    def test_return_chips_when_overbet(self):
        game = Game(4,[200,160,120,20])
        game.new_hand(first_hand=True)
        game.deck[8:13] = [Card(14,2),Card(14,3),Card(13,2),Card(13,3),Card(2,0)]
        cards_120 = [Card(3,0),Card(4,1)]
        cards_200 =[Card(13,0),Card(13,1)]
        game.positions[0].holecards=cards_200
        game.positions[2].holecards= cards_120
        game.implement_action(game.next[-1],2,10)
        game.implement_action(game.next[-1],0)
        game.implement_action(game.next[-1],2,200)
        game.implement_action(game.next[-1],0)
        game.implement_action(game.next[-1],1)
        self.assertTrue(game.finished)
        self.assertEqual(game.positions[0].stack,322)
        self.assertEqual(game.positions[1].stack,158)
        self.assertEqual(game.positions[2].stack,0)
        self.assertEqual(game.positions[3].stack,20)


    def test_return_chips_when_overbet_multi(self):
        game = Game(5,[200,160,120,20,150])
        game.new_hand(first_hand=True)
        game.deck[10:15] = [Card(14,2),Card(14,3),Card(13,2),Card(13,3),Card(2,0)]
        cards_120 = [Card(14,0),Card(14,1)]
        cards_150 = [Card(13,0),Card(13,1)]
        cards_200 = [Card(3,0),Card(4,1)]
        game.positions[0].holecards=cards_200
        game.positions[2].holecards=cards_120
        game.positions[4].holecards= cards_150
        game.implement_action(game.next[-1],2,10)
        game.implement_action(game.next[-1],0)
        game.implement_action(game.next[-1],2,80)
        game.implement_action(game.next[-1],1)
        game.implement_action(game.next[-1],0)
        game.implement_action(game.next[-1],2,120)
        game.implement_action(game.next[-1],1)
        game.implement_action(game.next[-1],1)
        game.implement_action(game.next[-1],1)
        game.implement_action(game.next[-1],2,30)
        game.implement_action(game.next[-1],1)
        self.assertTrue(game.finished)
        self.assertEqual(game.positions[0].stack,50)
        self.assertEqual(game.positions[1].stack,158)
        self.assertEqual(game.positions[2].stack,362)
        self.assertEqual(game.positions[3].stack,20)
        self.assertEqual(game.positions[4].stack,60)



    #side pots
    def test_side_pots(self):
        game = Game(4,[200,160,120,20])
        game.new_hand(first_hand=True)
        game.deck[8:13] = [Card(14,2),Card(14,3),Card(13,2),Card(13,3),Card(2,0)]
        cards_20 = [Card(14,0),Card(14,1)]
        cards_120 = [Card(13,0),Card(13,1)]
        cards_160 = [Card(2,1),Card(2,2)]
        cards_200 = [Card(3,0),Card(4,1)]
        game.positions[0].holecards=cards_200
        game.positions[1].holecards=cards_160
        game.positions[2].holecards= cards_120
        game.positions[3].holecards=cards_20
        game.implement_action(game.positions[2],2,120)
        game.implement_action(game.positions[3],1)
        game.implement_action(game.positions[0],2,200)
        game.implement_action(game.positions[1],1)
        self.assertTrue(game.finished)
        self.assertEqual(game.positions[3].stack,80)
        self.assertEqual(game.positions[2].stack,300)
        self.assertEqual(game.positions[1].stack,80)
        self.assertEqual(game.positions[0].stack,40)

    #side pots with split
    def test_side_pots_split(self):
        game = Game(4,[200,160,120,20])
        game.new_hand(first_hand=True)
        game.deck[8:13] = [Card(14,2),Card(14,3),Card(13,2),Card(13,3),Card(2,0)]
        cards_20 = [Card(2,1),Card(3,1)]
        cards_120 = [Card(14,0),Card(13,1)]
        cards_160 = [Card(14,1),Card(13,0)]
        cards_200 = [Card(2,2),Card(2,3)]
        game.positions[0].holecards=cards_200
        game.positions[1].holecards=cards_160
        game.positions[2].holecards= cards_120
        game.positions[3].holecards=cards_20
        game.implement_action(game.positions[2],2,120)
        game.implement_action(game.positions[3],1)
        game.implement_action(game.positions[0],2,200)
        game.implement_action(game.positions[1],1)
        self.assertTrue(game.finished)
        self.assertEqual(game.positions[3].stack,0)
        self.assertEqual(game.positions[2].stack,190)
        self.assertEqual(game.positions[1].stack,270)
        self.assertEqual(game.positions[0].stack,40)

    def test_side_pots_split_complex(self):
        game = Game(9,[200,160,120,20, 200,200,200,200,200])
        game.new_hand(first_hand=True)
        game.deck[18:23] = [Card(14,2),Card(14,3),Card(13,2),Card(13,3),Card(2,0)]
        cards_20 = [Card(2,1),Card(3,1)]
        cards_120 = [Card(14,0),Card(13,1)]
        cards_160 = [Card(14,1),Card(13,0)]
        cards_200 = [Card(2,2),Card(2,3)]
        cards_200_2 = [Card(4,2),Card(4,3)]
        game.positions[0].holecards=cards_200
        game.positions[1].holecards=cards_160
        game.positions[2].holecards= cards_120
        game.positions[3].holecards=cards_20
        game.positions[8].holecards=cards_200_2
        game.implement_action(game.positions[2],2,120)
        game.implement_action(game.positions[3],1)
        game.implement_action(game.positions[4],1)
        game.implement_action(game.positions[5],1)
        game.implement_action(game.positions[6],1)
        game.implement_action(game.positions[7],1)
        game.implement_action(game.positions[8],1)
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[1],1)
        game.implement_action(game.positions[0],2,40)
        game.implement_action(game.positions[1],1)
        game.implement_action(game.positions[4],0)
        game.implement_action(game.positions[5],0)
        game.implement_action(game.positions[6],0)
        game.implement_action(game.positions[7],1)
        game.implement_action(game.positions[8],1)
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[7],2,20)
        game.implement_action(game.positions[8],2,40)
        game.implement_action(game.positions[0],1)
        game.implement_action(game.positions[7],0)

        self.assertTrue(game.finished)
        self.assertEqual(game.positions[3].stack,0)
        self.assertEqual(game.positions[2].stack,490)
        self.assertEqual(game.positions[1].stack,650)
        self.assertEqual(game.positions[0].stack,100)

    def test_buggy_bet_sequence(self):
        # some betsequences lead to game.added becomming negative
        game = Game(2, [10, 10])
        game.new_hand(first_hand=True)
        game.implement_action(game.next[-1], 1)
        game.implement_action(game.next[-1], 1)
        game.implement_action(game.next[-1], 1)
        game.implement_action(game.next[-1], 2, 3)
        game.implement_action(game.next[-1], 2, 6)
        legal_sizes = game.get_legal_betsize()
        self.assertEqual(legal_sizes, 8)
        game.implement_action(game.next[-1], 2, 8)
