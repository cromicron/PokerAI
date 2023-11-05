import random
import copy
class PokerSimple:
    def __init__(self,agent_0, agent_1, multiple_run = True, run_ranges = False, ranges = {0:None, 1:None}): #If allin happened preflop, then we can let all runs happen (3 cards left)
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        self.deck = list(range(1,6))
        self.deck.extend(list(range(1,6)))
        self.multiple_run = multiple_run
        self.run_ranges = run_ranges
        self.ranges = ranges
    def reset(self):
        self.done = False
        self.position_0 = random.randint(0,1)
        self.position_1 = 0 if self.position_0 ==1 else 1
        self.street = 0
        self.pot = 1.5
        self.history = [] #overall 10 actions are possible
        if self.position_0 == 0:
            self.stacks = [4.5,4]

        else:
            self.stacks = [4, 4.5]
        random.shuffle(self.deck)
        self.hole_0 = self.deck[0]
        self.hole_1 = self.deck[1]
        self.board = 0 #no card dealt yet
        self.next_to_act = [0, 1] if self.position_0 == 0 else [1, 0]
        self.observations= [[],[]]
        self.create_observation(self.next_to_act[0])
        return self.done, self.next_to_act[0], self.observations[self.next_to_act[0]]
    
    def create_observation(self,player = 0):
        if player == 0:
            self.observations[0] = [self.position_0, self.stacks[0], self.stacks[1], self.street, self.hole_0, self.pot, self.board]
            history_obs = self.history[:]
            for i in range(10-len(self.history)):
                history_obs.append(-1)
            self.observations[0].extend(history_obs)
            
        else:
            self.observations[1] = [self.position_1, self.stacks[1], self.stacks[0], self.street, self.hole_1, self.pot, self.board]
            history_obs = self.history[:]
            for i in range(10-len(self.history)):
                history_obs.append(-1)
            self.observations[1].extend(history_obs)
    
    def showdown(self):
        win_0 = 0
        win_1 = 0
        split = 0
        if self.board == 0:
            if self.multiple_run:
                run = 8
            else:
                run = 1
        else:
            run = 1
        self.hand_history=[]
        for i in range(run):
            board = self.deck[2+i]

            if self.hole_0 == board:
                strength_0 = 10
            else:
                strength_0 = self.hole_0

            if self.hole_1 == board:
                strength_1 = 10
            else:
                strength_1 = self.hole_1

            if strength_0 > strength_1:
                win_0 += 1
            elif strength_1 > strength_0:
                win_1 += 1
            else:
                split += 1

            hand = [self.hole_0, self.hole_1, board]
            self.hand_history.append(hand)
        
        return win_0, win_1, split
        
    
    def implement_action(self, player, action):
        #actions: 0 = fold or check, depending on whether pot is balanced or not
        #         1 = check or call, depending on whether pot is balanced or not
        #         2 = allin or call, depending on whether pot is balanced or not
        if self.done:
            raise  RuntimeError('Poker Round is finished. Cannot implement any more actions. Start new hand!')
            return 
        if player != self.next_to_act[0]:
            raise  RuntimeError('{} is not next to act. Cannot implement action for this player'.format(player))
            return
       
        if self.stacks[0] == self.stacks[1]:
            if action == 0: #transform fold to check
                action = 1
        if self.stacks[0] == 0 or self.stacks[1] == 0: #one player is allin
            if action ==2: #action can only be call
                action =1
            
        if action == 0:
            if player == 0:
                self.stacks[1] += self.pot
            else:
                self.stacks[0] += self.pot
            bet = 0
            self.done=True
        
        elif action == 1: #call or check

            if len(self.next_to_act)==1: #last to act

                if player == 0:
                    bet = self.stacks[0]
                    self.pot += self.stacks[0] - self.stacks[1]
                    self.stacks[0] = self.stacks[1]
                    bet -= self.stacks[0] #bet is difference between stack before action and stack after action
                else:
                    bet = self.stacks[1]
                    self.pot += self.stacks[1] - self.stacks[0]
                    self.stacks[1] = self.stacks[0]
                    bet -= self.stacks[1]
                    
                if self.street == 0:

                    if self.stacks[0] != 0:
                        self.board = self.deck[2]
                        self.street = 1
                        if player == 1:
                            self.next_to_act =[1,0]
                        else:
                            self.next_to_act =[0,1]
                    else:
                        self.next_to_act = []
                
                else:
                    self.next_to_act = []
                    self.done = True

            else: #other player behind.
                #preflop first in:
                if self.street == 0 and self.pot == 1.5:
                    bet = 0.5
                    self.stacks[player] = 4
                    self.pot = 2
                    self.next_to_act.pop(0)
                
                else:
                    if player == 0:
                        self.next_to_act = [1]
                    else:
                        self.next_to_act = [0]
                    bet = 0
                
            if self.next_to_act == []:
                if not self.run_ranges:
                    win_0, win_1, split = self.showdown()
                else:
                    #showdown against entire range

                    for player in self.ranges:
                        if self.ranges[player] != None:
                            range_player = player

                    n_run = 20
                    win_0, win_1, split = 0, 0, 0
                    for i in range(n_run):
                        if range_player == 0:
                            hand_villain = self.hole_0 #just for memories
                            #sake. We are not using the actual hand of villain,
                            #because it is unknown.
                        else:
                            hand_villain = self.hole_1

                        game = copy.deepcopy(self)

                        #draw new hand_villain from range
                        new_hand_vil = random.choices([1,2,3,4,5], self.ranges[player])[0]
                        #exchange cards for villain
                        if new_hand_vil != hand_villain:
                            index_new_hand_vil = game.deck[2:].index(new_hand_vil)+2
                            if range_player == 0:
                                game.hole_0 = new_hand_vil
                                game.deck[index_new_hand_vil], game.deck[1] = hand_villain, new_hand_vil
                            else:
                                game.hole_1 = new_hand_vil
                                game.deck[index_new_hand_vil], game.deck[0] = hand_villain, new_hand_vil
                        w_0, w_1, s = game.showdown()
                        win_0 += w_0
                        win_1 += w_1
                        split += s
                    win_0 /= n_run
                    win_1 /= n_run
                    split /= n_run
                proportion_pot_0 = win_0/(win_0+win_1+split)+0.5*(split/(win_0+win_1+split))
                proportion_pot_1 = 1 - proportion_pot_0
                self.stacks[0] += proportion_pot_0 * self.pot
                self.stacks[1] += proportion_pot_1 * self.pot
                self.done = True

                    
        else: #action allin.
            if player == 0:
                bet = self.stacks[0]
                self.pot += self.stacks[0]
                self.stacks[0] = 0
                self.next_to_act = [1]
                bet -= self.stacks[0]
            else:
                bet = self.stacks[1]
                self.pot += self.stacks[1]
                self.stacks[1] = 0
                self.next_to_act = [0]
                bet-= self.stacks[1]
                
        self.history.append(bet)