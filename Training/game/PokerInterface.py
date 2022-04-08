from PokerGame import HUPoker
#from Agent import Agent

class PokerInterface():
    def __init__(self, hero, villain, stack_hero=100, stack_villain=100):
        self.hero = hero
        self.villain = villain
        self.poker = HUPoker(stack_hero, stack_villain, self.hero, self.villain, n_run =1)


    def simulate(self):
        poker = self.poker
        observation = poker.reset()
        hand = poker.holecards
        pos_player = ['hero','villain'] if poker.position == 0 else ['villain','hero']

        value_dict = {10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        for i in range (2,10):
            value_dict[i] = str(i)
        suit_dict = {0 : '\u2660', 1:'\u2665', 2: '\u2663', 3: '\u2666'} #creating suits from unicode
        dict_action = {0: 'folds', 1: 'checks/calls', 2: 'minbets/raises', 3: 'bets/raises', 4: 'bets/raises', 5: 'bets/raises', 6: 'bets/raises', 7: 'bets/raises',
                       8: 'bets/raises', 9: 'bets/raises', 10: 'bets/raises', 11: 'goes all in',}

        card1 = value_dict[hand[0][0]]+suit_dict[hand[0][1]]
        card2 = value_dict[hand[1][0]]+suit_dict[hand[1][1]]
        print(pos_player[0], "is the button")            
        print("Stack Hero ", poker.starting_hero)
        print("Stack Villain", poker.starting_villain)
        print(pos_player[0], 'posts small blind 0.5')
        print(pos_player[1], 'posts big blind 1')
        print("*** HOLE CARDS ***")
        print("Dealt to Hero ", card1, card2)

        done = False
        if poker.position == 1:
            action = poker.action_villain
            print('Villain', dict_action[action])
            if action == 0:
                done = True
            poker.left_bet_round = [1]

        next_to_act = 'hero'            


        while not done:
            if poker.bet_sb == poker.bet_bb:
                next_to_act = 'hero' if poker.position == 1 else 'villain'
            street = poker.street
            if street != 0:
                print("Stack Hero ", poker.stack_sb if poker.position == 0 else poker.stack_bb)
                print("Stack Villain ", poker.stack_sb if poker.position == 1 else poker.stack_bb)
                board = ""
                for i in range(len(poker.board_current)):
                            board += value_dict[poker.board_current[i][0]]+suit_dict[poker.board_current[i][1]]
                if street == 1:
                    print("*** FLOP ***", board)

                elif street == 2:
                    print("*** TURN ***", board[:-2], " ", board[-2:])

                else:
                    print("*** RIVER ***", board[:-2], " ", board[-2:])

            while street == poker.street and not done:
                if next_to_act == 'hero':
                    poker.create_observation()
                    observation = poker.observation
                    action = self.hero.choose_action(observation)

                else:

                    observation_villain = poker.create_observation_villain()
                    action = self.villain.choose_action(observation_villain)


                done = poker.implement_action(action, player = next_to_act)
                print(next_to_act, action, dict_action[action])
                next_to_act = 'villain' if next_to_act == 'hero' else 'hero'

        if action != 0: #if noone folded
            print("*** SHOW DOWN ***")
            card1_villain = value_dict[poker.holecards_villain[0][0]]+suit_dict[poker.holecards_villain[0][1]]
            card2_villain = value_dict[poker.holecards_villain[1][0]]+suit_dict[poker.holecards_villain[1][1]]
            if poker.position == 0:
                print(pos_player[0], "shows", card1, card2 )
                print(pos_player[1], "shows", card1_villain, card2_villain)
            else:
                print(pos_player[0], "shows", card1_villain, card2_villain)
                print(pos_player[1], "shows", card1, card2 )

            board = ""
            for i in range(len(poker.board_current)):
                        board += value_dict[poker.board_current[i][0]]+suit_dict[poker.board_current[i][1]]
            if street < 1:
                print("*** FLOP ***", board[:6])

            if street < 2:
                print("*** TURN ***", board[:6], " ", board[6:-2])

            if street < 3:
                print("*** RIVER ***", board[:-2], " ", board[-2:])

        if (poker.stack_sb > poker.starting_hero and poker.position == 0) or poker.stack_bb > poker.starting_hero and poker.position == 1:
            print("Hero wins", poker.pot)
        elif poker.stack_sb == poker.starting_hero or poker.stack_bb == poker.starting_hero:
            print("Split pot. Bot players receive", poker.pot/2)
        else:
            print("Villain wins", poker.pot)

        print("*** SUMMARY ***")
        print("Difference stack", pos_player[0], poker.stack_sb - poker.starting_hero if poker.position == 0 else poker.stack_sb -poker.starting_villain)
        print("Difference stack", pos_player[1], poker.stack_bb - poker.starting_hero if poker.position == 1 else poker.stack_bb -poker.starting_villain)

def play_against(hero, villain,n_games = 1000):

    hero = hero
    villain = villain
    env = HUPoker(20,20, hero, villain, n_run=20)
    total_score = 0

    for i in range(n_games):
        observation, done = env.reset()
        reward = 0
                
        while not done:
            action = hero.choose_action(observation)
            done = env.step(action)
            observation = env.observation

       
                
        reward = env.stack_sb - env.starting_hero if env.position == 0 else env.stack_bb - env.starting_hero 

        total_score += reward
        avg_score = total_score/(i+0.00001)
        print('episode: ', i, 'reward %.2f' % reward,
                        'average_score %.2f' % avg_score,
                        'total score %.2f' % total_score)
    return avg_score, env.observation

