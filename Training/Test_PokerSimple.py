from game.PokerSimple import PokerSimple
import unittest
hero = 0
villain = 1
game = PokerSimple(hero, villain)
class TestPokerSimple(unittest.TestCase):


    def test_wrong_input(self):
        done, next_to_act, observation = game.reset()
        if next_to_act == 0:
            wrong_player = 1
        else:
            wrong_player = 0
        self.assertRaises(RuntimeError, game.implement_action, wrong_player, 0)
    
    def test_done(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 0)
        if next_to_act == 0:
            player = 1
        else:
            player = 0
        with self.assertRaises(Exception):
            game.implement_action(player, 0)                       
       
    def test_call_sb_first_in(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        game.create_observation(next_to_act)
        if next_to_act ==0:
            hole = game.hole_0
        else:
            hole = game.hole_1
        expected_observation = [0, 4, 4, 0, hole, 2, 0, 0.5, -1, -1, -1,-1,-1,-1,-1,-1,-1]
        self.assertEqual(game.observations[next_to_act], expected_observation)
        
    def test_raise_sb_first_in(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 2)
        game.create_observation(next_to_act)
        if next_to_act ==0:
            hole = game.hole_0
        else:
            hole = game.hole_1
        expected_observation = [0, 0, 4, 0, hole, 6, 0, 4.5, -1, -1, -1,-1,-1,-1,-1,-1,-1]
        self.assertEqual(game.observations[next_to_act], expected_observation)
        
    def test_first_in_called_to_bb(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        game.create_observation(game.next_to_act[0])
        if game.next_to_act[0] ==0:
            hole = game.hole_0
        else:
            hole = game.hole_1
        expected_observation = [1, 4, 4, 0, hole, 2, 0, 0.5, -1, -1, -1,-1,-1,-1,-1,-1,-1]
        self.assertEqual(game.observations[game.next_to_act[0]], expected_observation)

    def test_check_second_in_pf(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        game.create_observation(game.next_to_act[0])

        game.implement_action(game.next_to_act[0],1)
        if game.next_to_act[0] ==0:
            hole = game.hole_0
        else:
            hole = game.hole_1
        game.create_observation(game.next_to_act[0])
        expected_observation = [1, 4, 4, 1, hole, 2, game.board, 0.5, 0, -1, -1,-1,-1,-1,-1,-1,-1]
        self.assertEqual(game.observations[game.next_to_act[0]], expected_observation)
        
    def test_push_second_in_pf(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        game.create_observation(game.next_to_act[0])

        game.implement_action(game.next_to_act[0],2)
        if game.next_to_act[0] ==0:
            hole = game.hole_0
        else:
            hole = game.hole_1
        game.create_observation(game.next_to_act[0])
        expected_observation = [0, 4, 0, 0, hole, 6, 0, 0.5, 4, -1, -1,-1,-1,-1,-1,-1,-1]
        self.assertEqual(game.observations[game.next_to_act[0]], expected_observation)
            
    def test_showdown_preflop(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 2)
        game.implement_action(game.next_to_act[0],1)
        game.create_observation(0)
        game.create_observation(1)
        obs_0, obs_1 = game.observations
        self.assertEqual(game.observations[0][1], game.observations[1][2])
        self.assertEqual(len(game.next_to_act),0)
        self.assertTrue(game.done)

    #test all possible combinations of actions
    def test_fold_sb_first_in(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 0)
        game.create_observation(1)
        game.create_observation(0)
            
        if next_to_act ==0:
            hole_sb = game.hole_0
            hole_bb = game.hole_1
            
        else:
            hole_sb = game.hole_1
            hole_bb = game.hole_0

        expected_observation_sb = [0, 4.5, 5.5, 0, hole_sb, 1.5, 0, 0, -1, -1, -1,-1,-1,-1,-1,-1,-1]
        expected_observation_bb = [1, 5.5, 4.5, 0, hole_bb, 1.5, 0, 0, -1, -1, -1,-1,-1,-1,-1,-1,-1]
        
        
        self.assertTrue(game.done, 'when the player folds, done must be true')
        self.assertEqual(game.observations[next_to_act], expected_observation_sb)
        bb = 0 if next_to_act == 1 else 1
        self.assertEqual(game.observations[bb], expected_observation_bb)
        
        
    
    #sb calls, bb checks, (flop), bb checks, sb checks
    def test_showdown_postflop_checkdown(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        game.implement_action(game.next_to_act[0],1)
        game.implement_action(game.next_to_act[0], 1)
        game.implement_action(game.next_to_act[0], 1)
        game.create_observation(0)
        game.create_observation(1)
        obs_0, obs_1 = game.observations
        if obs_0[4] == obs_0[6] and obs_1[4]!= obs_1[6]:
            winner = 0
        elif obs_1[4] == obs_1[6] and obs_0[4]!= obs_0[6]:
            winner = 1
        elif obs_0[4] == obs_0[6] and obs_1[4]== obs_1[6]:
            winner = 2
        elif obs_0[4] > obs_1[4]:
            winner = 0
        elif obs_0[4] < obs_1[4]:
            winner = 1
        else:
            winner =2
        
        if winner == 0:
            self.assertEqual(game.observations[0][1], 6)
        elif winner == 1:
            self.assertEqual(game.observations[1][1], 6)
        else:
            self.assertEqual(game.observations[0][1], 5)
            
        self.assertTrue(game.done)    
    
    #sb calls, bb checks, (flop), bb checks, sb allin, bb folds
    def test_sb_allin_flop(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,1)
        game.implement_action(next_to_act, 1)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,2)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act, 0)
        
        game.create_observation(next_to_act)
        sb = 0 if next_to_act ==1 else 1
        game.create_observation(sb)
        hole_sb = game.hole_0 if next_to_act ==1 else game.hole_1
        hole_bb = game.hole_1 if next_to_act ==1 else game.hole_0
        
        expected_observation_sb = [0, 6, 4, 1, hole_sb, 6, game.board, 0.5, 0, 0, 4,0,-1,-1,-1,-1,-1]
        expected_observation_bb = [1, 4, 6, 1, hole_bb, 6, game.board, 0.5, 0, 0, 4,0,-1,-1,-1,-1,-1]
        
        observation_sb = game.observations[sb]
        observation_bb = game.observations[next_to_act]
        
        self.assertEqual(observation_sb, expected_observation_sb)
        self.assertEqual(observation_bb, expected_observation_bb)
        self.assertTrue(game.done) 
        
    #sb calls, bb checks, (flop), bb checks, sb allin, bb calls
    def test_sb_allin_flop_bbcall(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,1)
        game.implement_action(next_to_act, 1)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,2)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act, 1)
        
        game.create_observation(next_to_act)
        sb = 0 if next_to_act ==1 else 1
        game.create_observation(sb)
        hole_sb = game.hole_0 if next_to_act ==1 else game.hole_1
        hole_bb = game.hole_1 if next_to_act ==1 else game.hole_0
        
        if (hole_sb == game.board and hole_bb != game.board) or (hole_bb != game.board and hole_sb > hole_bb):
            winner = sb
        elif (hole_bb == game.board and hole_sb != game.board) or (hole_sb != game.board and hole_bb > hole_sb):
            winner = next_to_act
        else:
            winner = 2
        
        if winner == sb:
            expected_observation_sb = [0, 10, 0, 1, hole_sb, 10, game.board, 0.5, 0, 0, 4, 4,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, 0, 10, 1, hole_bb, 10, game.board, 0.5, 0, 0, 4, 4,-1,-1,-1,-1,-1]
            
        elif winner == next_to_act:
            expected_observation_sb = [0, 0, 10, 1, hole_sb, 10, game.board, 0.5, 0, 0, 4, 4,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, 10, 0, 1, hole_bb, 10, game.board, 0.5, 0, 0, 4, 4,-1,-1,-1,-1,-1]
            
        else:
            expected_observation_sb = [0, 5, 5, 1, hole_sb, 10, game.board, 0.5, 0, 0, 4, 4,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, 5, 5, 1, hole_bb, 10, game.board, 0.5, 0, 0, 4, 4,-1,-1,-1,-1,-1]
            
        observation_sb = game.observations[sb]
        observation_bb = game.observations[next_to_act]
        
        self.assertEqual(observation_sb, expected_observation_sb)
        self.assertEqual(observation_bb, expected_observation_bb)
        self.assertTrue(game.done) 
    #sb calls, bb checks, (flop), bb allin, sb folds
    def test_bb_allin_sb_folds_flop(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,1)
        game.implement_action(next_to_act, 2)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,0)
        
        game.create_observation(next_to_act)
        bb = 0 if next_to_act ==1 else 1
        game.create_observation(bb)
        hole_sb = game.hole_0 if next_to_act ==0 else game.hole_1
        hole_bb = game.hole_1 if next_to_act ==0 else game.hole_0
        
        expected_observation_sb = [0, 4, 6, 1, hole_sb, 6, game.board, 0.5, 0, 4, 0, -1,-1,-1,-1,-1,-1]
        expected_observation_bb = [1, 6, 4, 1, hole_bb, 6, game.board, 0.5, 0, 4, 0, -1,-1,-1,-1,-1,-1]
            
        observation_sb = game.observations[next_to_act]
        observation_bb = game.observations[bb]
        
        self.assertEqual(observation_sb, expected_observation_sb)
        self.assertEqual(observation_bb, expected_observation_bb)
        self.assertTrue(game.done) 
    #sb calls, bb checks, (flop), bb allin, sb calls
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,1)
        game.implement_action(next_to_act, 2)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,1)
        
        game.create_observation(next_to_act)
        bb = 0 if next_to_act ==1 else 1
        game.create_observation(bb)
        hole_sb = game.hole_0 if next_to_act ==0 else game.hole_1
        hole_bb = game.hole_1 if next_to_act ==0 else game.hole_0
        
        if (hole_sb == game.board and hole_bb != game.board) or (hole_bb != game.board and hole_sb > hole_bb):
            winner = next_to_act
        elif (hole_bb == game.board and hole_sb != game.board) or (hole_sb != game.board and hole_bb > hole_sb):
            winner = bb
        else:
            winner = 2
        
        if winner == next_to_act:
            expected_observation_sb = [0, 10, 0, 1, hole_sb, 10, game.board, 0.5, 0, 4, 4, -1,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, 0, 10, 1, hole_bb, 10, game.board, 0.5, 0, 4, 4, -1,-1,-1,-1,-1,-1]
            
        elif winner == bb:
            expected_observation_sb = [0, 0, 10, 1, hole_sb, 10, game.board, 0.5, 0, 4, 4, -1,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, 10, 0, 1, hole_bb, 10, game.board, 0.5, 0, 4, 4, -1,-1,-1,-1,-1,-1]
            
        else:
            expected_observation_sb = [0, 5, 5, 1, hole_sb, 10, game.board, 0.5, 0, 4, 4, -1,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, 5, 5, 1, hole_bb, 10, game.board, 0.5, 0, 4, 4, -1,-1,-1,-1,-1,-1]
            
        observation_sb = game.observations[next_to_act]
        observation_bb = game.observations[bb]
        
        self.assertEqual(observation_sb, expected_observation_sb)
        self.assertEqual(observation_bb, expected_observation_bb)
        self.assertTrue(game.done) 
    #sb calls, bb allin, sb folds
    def test_bb_allin_sb_folds_preflop(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,2)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act, 0)
        
        game.create_observation(next_to_act)
        bb = 0 if next_to_act ==1 else 1
        game.create_observation(bb)
        hole_sb = game.hole_0 if next_to_act ==0 else game.hole_1
        hole_bb = game.hole_1 if next_to_act ==0 else game.hole_0
        
        expected_observation_sb = [0, 4, 6, 0, hole_sb, 6, 0, 0.5, 4, 0, -1, -1,-1,-1,-1,-1,-1]
        expected_observation_bb = [1, 6, 4, 0, hole_bb, 6, 0, 0.5, 4, 0, -1, -1,-1,-1,-1,-1,-1]
            
        observation_sb = game.observations[next_to_act]
        observation_bb = game.observations[bb]
        
        self.assertEqual(observation_sb, expected_observation_sb)
        self.assertEqual(observation_bb, expected_observation_bb)
        self.assertTrue(game.done) 
    #sb calls, bb allin, sb calls
    def test_bb_allin_sb_calls_preflop(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 1)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,2)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act, 1)
        count, nwin_0, nwin_1, nsplit = 0,0,0,0
        for hand in game.hand_history:
            count += 1
            if (hand[0] == hand[2] and hand [1] != hand[2]) or (hand[1] != hand[2] and hand[0] > hand[1]):
                nwin_0+=1
            elif (hand[1] == hand[2] and hand [0] != hand[2]) or (hand[0] != hand[2] and hand[1] > hand[0]):
                nwin_1 += 1
            else:
                nsplit += 1
        expected_stack_0 = (nwin_0 + 0.5*nsplit)*10/count
        expected_stack_1 = (nwin_1 + 0.5*nsplit)*10/count
        game.create_observation(0)
        game.create_observation(1)
        if next_to_act == 0: # sb is 0
            expected_observation_sb = [0, expected_stack_0, expected_stack_1, 0, hand[0], 10, 0, 0.5, 4, 4, -1, -1,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, expected_stack_1, expected_stack_0, 0, hand[1], 10, 0, 0.5, 4, 4, -1, -1,-1,-1,-1,-1,-1]
            observation_sb = game.observations[next_to_act]
            observation_bb = game.observations[1]
        else:
            expected_observation_sb = [0, expected_stack_1, expected_stack_0, 0, hand[1], 10, 0, 0.5, 4, 4, -1, -1,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, expected_stack_0, expected_stack_1, 0, hand[0], 10, 0, 0.5, 4, 4, -1, -1,-1,-1,-1,-1,-1]
            observation_sb = game.observations[next_to_act]
            observation_bb = game.observations[0]
        self.assertEqual(observation_sb, expected_observation_sb)
        self.assertEqual(observation_bb, expected_observation_bb)
        self.assertTrue(game.done) 
            
                
    #sb allin, bb folds
    def test_sb_allin_bb_folds_preflop(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 2)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act,0)
        game.create_observation(0)
        game.create_observation(1)
        sb = 0 if next_to_act ==1 else 1
        hole_sb = game.hole_0 if next_to_act == 1 else game.hole_1
        hole_bb = game.hole_0 if next_to_act == 0 else game.hole_1
        
        expected_observation_sb = [0, 6, 4, 0, hole_sb, 6, 0, 4.5, 0, -1, -1, -1,-1,-1,-1,-1,-1]
        expected_observation_bb = [1, 4, 6, 0, hole_bb, 6, 0, 4.5, 0, -1, -1, -1,-1,-1,-1,-1,-1]
        observation_sb = game.observations[sb]
        observation_bb = game.observations[next_to_act]
        self.assertEqual(observation_sb, expected_observation_sb)
        self.assertEqual(observation_bb, expected_observation_bb)
        self.assertTrue(game.done) 
    #sb allin, bb calls
    def test_sb_allin_bb_calls_preflop(self):
        done, next_to_act, observation = game.reset()
        game.implement_action(next_to_act, 2)
        next_to_act = 0 if next_to_act ==1 else 1
        game.implement_action(next_to_act, 1)
        game.create_observation(0)
        game.create_observation(1)

        count, nwin_0, nwin_1, nsplit = 0,0,0,0
        for hand in game.hand_history:
            count += 1
            if (hand[0] == hand[2] and hand [1] != hand[2]) or (hand[1] != hand[2] and hand[0] > hand[1]):
                nwin_0+=1
            elif (hand[1] == hand[2] and hand [0] != hand[2]) or (hand[0] != hand[2] and hand[1] > hand[0]):
                nwin_1 += 1
            else:
                nsplit += 1
        expected_stack_0 = (nwin_0 + 0.5*nsplit)*10/count
        expected_stack_1 = (nwin_1 + 0.5*nsplit)*10/count
        game.create_observation(0)
        game.create_observation(1)
        if next_to_act == 1: # sb is 0
            expected_observation_sb = [0, expected_stack_0, expected_stack_1, 0, hand[0], 10, 0, 4.5, 4, -1, -1, -1,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, expected_stack_1, expected_stack_0, 0, hand[1], 10, 0, 4.5, 4, -1, -1, -1,-1,-1,-1,-1,-1]
            observation_sb = game.observations[0]
            observation_bb = game.observations[1]
        else:
            expected_observation_sb = [0, expected_stack_1, expected_stack_0, 0, hand[1], 10, 0, 4.5, 4, -1, -1, -1,-1,-1,-1,-1,-1]
            expected_observation_bb = [1, expected_stack_0, expected_stack_1, 0, hand[0], 10, 0, 4.5, 4, -1, -1, -1,-1,-1,-1,-1,-1]
            observation_sb = game.observations[1]
            observation_bb = game.observations[0]
            
        self.assertEqual(observation_sb, expected_observation_sb)
        self.assertEqual(observation_bb, expected_observation_bb)
        self.assertTrue(game.done) 

unittest.main()
    