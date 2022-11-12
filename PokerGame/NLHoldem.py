import random
import collections
from HandComperator import compare_hands
import copy
import numpy as np

value_dict = {10: 'T', 11: 'J', 12: 'Q', 13:'K', 14: 'A'}
suit_dict = {0:'c', 1:'d', 2:'h', 3:'s'}
class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
        self.representation = str(value)+suit_dict[suit] if value < 10 else value_dict[value]+suit_dict[suit]

class Deck(list):
    def __init__(self):
        super().__init__()
        for suit in range(4):
            for value in range(2,15):
                new_card = Card(value, suit)
                self.append(new_card)

    def shuffle(self):
        random.shuffle(self)


class Player:
    def __init__(self, name, stack=100):
        self.name = name
        self.stack = stack
        self.starting_stack = copy.copy(stack)
        self.holecards = None
        self.bet = 0 #how much has player put into the pot in the current betting round
    def get_holecard_representation(self):
        return self.holecards[0].representation+self.holecards[1].representation

class Pot:
    def __init__(self, players, pot_size):
        self.players = players
        self.pot_size = pot_size

class SizeError(Exception):
    pass

class Hand_History:
    def __init__(self):
        pass

class Game:
    def __init__(self, n_players=2, stacks=[], hand_history = False):
        self.n_players = n_players
        self.deck = Deck()
        self.players = []
        self.max_bet = 0 #max_bet current round
        if len(stacks) != n_players and stacks != []:
            raise ArithmeticError("number of stacks must equal number of players")

        if stacks == []:
            self.starting_stacks = [100 for player in range(n_players)]
        else:
            self.starting_stacks = stacks
        if min(self.starting_stacks) < 2:
            raise ValueError("Starting Stack has to be at least 2")

        for i in range(n_players):
            new_player = Player(i, self.starting_stacks[i])
            self.players.append(new_player)
        self.positions = collections.deque(self.players)
        self.board = []
        self.pot = 0
        if hand_history:
            self.hand_history = Hand_History()

    def new_hand(self, random_seat=False, first_hand = False):
        self.left_in_hand = copy.copy(self.positions)
        self.deck.shuffle()
        if random_seat:
            random.shuffle(self.positions)
        else:
            if not first_hand:
                self.positions.rotate(-1)

        for i in range(len(self.positions)):
            self.positions[i].holecards=self.deck[2*i:2*i+2]
        self.positions[0].stack -= 0.5
        self.positions[0].bet += 0.5
        self.positions[1].stack -= 1
        self.positions[1].bet += 1
        self.pot = 1.5
        self.next = collections.deque(self.positions) #the next player to act is on top of the que
        self.next.reverse()
        if self.n_players >2:
            self.next.rotate(2)
        self.roundabout = copy.copy(self.next)
        self.street = 0
        self.finished = False
        self.max_bet =1
        self.added = 1 #current amount added to previous max_bet. Necessary to know minbet
    def compare(self,players):
        #takes in players and returns relative hand strengths as list
        hands = [self.board + [cards for cards in player.holecards] for player in players]
        strengths =  compare_hands(hands)
        return strengths

    def showdown(self, n_run = 1):
        #compares hands of remaining players and distributes chips accordingly
        #check for side pot possibilities: When a player is all in and has less chips than any of the remaining
        #players starting stacks
        result = np.array(self.compare(self.left_in_hand))
        if len(self.left_in_hand)<3 or 0 not in [player.stack for player in self.left_in_hand]: #two players can't have a side pot
            winners = np.flatnonzero(result == np.max(result))
            for winner in winners:
                self.left_in_hand[winner].stack += self.pot/len(winners)
        #for side-pots

        elif True:
            #create pots
            all_ins = []
            pots = []
            for player in self.left_in_hand:
                if player.stack == 0:
                    all_ins.append(player)
            all_ins.sort(key = lambda x: x.starting_stack)
            bets_folded = []
            for player in self.players:
                if player not in self.left_in_hand and player.starting_stack-player.stack >0:
                    bets_folded.append(player.starting_stack-player.stack)
            print(len(self.left_in_hand), len(self.players))
            #add all players who have equal or more than all in player to new pot
            previous_smallest = 0
            strengths = dict(zip(self.left_in_hand,result))
            print(bets_folded)
            while len(all_ins) > 0:
                current_smallest = all_ins[0].starting_stack
                pot = len(self.left_in_hand)*(current_smallest-previous_smallest)
                print("1",len(self.left_in_hand),pot)


                for i in range(len(bets_folded)):
                    to_add = min(bets_folded[i], current_smallest-previous_smallest)
                    print("to add",to_add)
                    pot += to_add
                    bets_folded[i] -= to_add
                previous_smallest = current_smallest
                print("2",pot)
                pots.append(Pot(copy.copy(self.left_in_hand),copy.copy(pot)))

                stack = current_smallest
                while len(all_ins)>0 and stack == all_ins[0].starting_stack:
                    self.left_in_hand.remove(all_ins[0])
                    all_ins.remove(all_ins[0])

            for pot in pots:
                winners = []
                to_remove = list(set(strengths.keys())-set(pot.players))
                for player in to_remove:
                    del(strengths[player])

                m = max(strengths,key = strengths.get)
                for player in pot.players:
                    if strengths[player] == strengths[m]:
                        winners.append(player)
                print(pot.pot_size,len(pot.players), len(strengths),winners)
                for winner in winners:
                    winner.stack += pot.pot_size/len(winners)

        else:
            #create list of all players who are allin
            strengths = dict(zip(self.left_in_hand,result))
            all_ins = []
            for player in self.left_in_hand:
                if player.stack == 0:
                    all_ins.append(player)
            all_ins.sort(key = lambda x: x.starting_stack)
            remove_from_pot = 0
            players_fold = list(set(self.players)-set(self.left_in_hand))
            players_fold_bet = []
            for player in players_fold:
                if player.bet != 0:
                    players_fold_bet.append(player)
            to_remove = 0
            for i in range(len(all_ins)):
                to_add = 0
                for player in players_fold_bet:
                    to_add += min([player.bet,all_ins[i].starting_stack-to_remove])
                    player.bet -= to_add
                to_remove +=all_ins[i].starting_stack
                pot = (all_ins[i].starting_stack-remove_from_pot)*len(self.left_in_hand)+to_add
                winners_pot = []
                for player in self.left_in_hand:
                    m = max(strengths,key = strengths.get)
                    if strengths[player] == strengths[m]:
                        winners_pot.append(player)
                for winner_pot in winners_pot:
                    winner_pot.stack += pot/len(winners_pot)
                print(all_ins[i].stack, all_ins[i].starting_stack)
                self.left_in_hand.remove(all_ins[i])
                del strengths[all_ins[i]]
                for k in range(i+1,len(all_ins)):
                    if all_ins[k].starting_stack == all_ins[i].starting_stack:
                        self.left_in_hand.remove(all_ins[k])
                        del strengths[all_ins[k]]
                remove_from_pot = all_ins[i].starting_stack

    def get_legal_actions(self):
        if self.finished:
            return None
        next_player = self.next[-1]
        if next_player.bet == self.pot/self.n_players:
            if next_player.stack != 0:
                return 1,2
            else:
                return 1

        else:
            if next_player.stack > self.max_bet - next_player.stack:
                return 0,1,2
            else:
                return 0,1

    def implement_action(self, player, action, amount=None):
        if player != self.next[-1]:
            raise ValueError("It's not the chosen player's turn to act")
            return
        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            errorMsg = str(action) + " is an illegal action. Please choose " + str(legal_actions)
            raise ValueError(errorMsg)
            return
        if amount != None and amount > player.stack+player.bet:
            raise SizeError("player doesn't have enough chips to make that bets/raise")
        if action == 0:
            self.next.pop()
            self.roundabout.pop()
            self.left_in_hand.remove(player)
            if len(self.left_in_hand)==1:
                #if after a fold, only one player is left, the round is finished and that player who is left, receives
                #the pot
                self.left_in_hand[0].stack+=self.pot
                self.next=[]
                self.finished=True

        elif action == 1:
            self.roundabout.rotate()
            self.next.pop()
            if player.bet != self.max_bet:
                #player calls
                #check if player has enough chips to call the maxbet
                to_call = self.max_bet - player.bet
                if player.stack < to_call:
                    self.pot += player.stack
                    player.bet += player.stack
                    player.stack = 0
                else:
                    self.pot += to_call
                    player.bet += to_call
                    player.stack -= to_call
            #remove from roundabout if no chips left
            if player.stack == 0:
                self.roundabout.remove(player)

        else: #player bets or raises. Everytime he does, he must spcify a bet/raise size.
            if amount == None:
                raise TypeError("Specify bet or raise size!")
                return
            #check for legal raise size. Size must be matching the current raise/bet and adding equal ammount
            #if raised, but only if player has enough.
            if (amount < player.bet+player.stack) and (amount < self.max_bet + self.added or amount < 1):
                 raise SizeError("illegal betsize!")
                 return
            #add all players, who aren't in next anymore back into next
            self.roundabout.rotate()
            for pl in reversed(self.roundabout):
                if pl not in self.next:
                    self.next.appendleft(pl)
            self.next.pop()
            self.pot += amount - player.bet
            player.stack -= amount - player.bet
            player.bet = copy.copy(amount)
            self.added = player.bet - self.max_bet
            self.max_bet = copy.copy(player.bet)
            #remove from roundabout if no chips left
            if player.stack == 0:
                self.roundabout.remove(player)

        #check if betting round is finished:
        if not self.finished:
            if len(self.next) == 0:
                #return overbets to player who bet more than could be called. One player must be all_in
                if 0 in [player.stack for player in self.left_in_hand]:
                    #find largest_raise
                    player_list = copy.copy(self.left_in_hand)
                    largest_bet = player_list[0]
                    for i in range(1,len(player_list)):
                        if player_list[i].bet > largest_bet.bet:
                            largest_bet = player_list[i]
                    player_list.remove(largest_bet)
                    largest_all_in = player_list[0]
                    for j in range(1,len(player_list)):
                        if player_list[j].bet > largest_all_in.bet:
                            largest_all_in = player_list[j]
                    too_much = largest_bet.bet - largest_all_in.bet
                    self.pot -= too_much
                    largest_bet.stack +=too_much


                #check if hand is finished
                if self.street == 3:
                    #showdown of all players
                    self.showdown()
                    self.finished = True

                else:
                    #check if there are at least two players with more than 0 chips
                    player_with_chips = 0
                    for player in self.left_in_hand:
                        if player.stack != 0:
                            player_with_chips += 1

                    if player_with_chips <= 1:
                        #no more possible actions, reveal board till river
                        if self.street == 0:
                            self.board.extend(self.deck[2*self.n_players:2*self.n_players +5])
                        elif self.street ==1:
                            self.board.extend(self.deck[2*self.n_players+3:2*self.n_players +5])
                        else:
                            self.board.extend(self.deck[2*self.n_players+4:2*self.n_players +5])

                        self.street = 3
                        self.showdown()
                        self.finished = True

                    else:
                        self.street += 1
                        if self.street == 1:
                            self.board.extend(self.deck[2*self.n_players:2*self.n_players +3])
                        else:
                            self.board.append(self.deck[2*self.n_players +self.street+1])
                        self.next = collections.deque(self.left_in_hand)

                        if self.n_players != 2:
                            self.next.reverse()
                        #remove all players with 0 chips from next
                        to_remove = []
                        for player in self.next:
                            if player.stack == 0:
                                to_remove.append(player)
                        for player in to_remove:
                            self.next.remove(player)
                        self.roundabout = copy.copy(self.next)
                        self.max_bet = 0
                        self.added = 0

                        for pl in self.left_in_hand:
                            pl.bet = 0

    def get_observation(self):
        if len(self.players)!= 2:
            raise ValueError("Observations are only implemented for Heads Up")
            return

        player = self.next[-1]
        position = self.positions.index(player)
        starting_stack_hero = player.starting_stack
        current_stack_hero = player.stack
        bet_street_hero = player.bet
        bet_total_hero = player.starting_stack - player.stack

        c_1_v = player.holecards[0].value
        c_1_s = player.holecards[0].suit
        c_2_v = player.holecards[1].value
        c_2_s = player.holecards[1].suit
        street = self.street
        pot_size = self.pot

        villain = self.positions[1-position]
        starting_stack_vil = villain.starting_stack
        current_stack_vil = villain.stack
        bet_street_vil = villain.bet
        bet_total_vil = starting_stack_vil - current_stack_vil
        if len(self.board)== 0:
            flop_1_v = -1
            flop_1_s = -1
            flop_2_v = -1
            flop_2_s = -1
            flop_3_v = -1
            flop_3_s = -1
            turn_v = -1
            turn_s = -1
            river_v = -1
            river_s = -1
        elif len(self.board)==3:
            flop_1_v = self.board[0].value
            flop_1_s = self.board[0].suit
            flop_2_v = self.board[1].value
            flop_2_s = self.board[1].suit
            flop_3_v = self.board[2].value
            flop_3_s = self.board[2].suit
            turn_v = -1
            turn_s = -1
            river_v = -1
            river_s = -1
        elif len(self.board)==4:
            flop_1_v = self.board[0].value
            flop_1_s = self.board[0].suit
            flop_2_v = self.board[1].value
            flop_2_s = self.board[1].suit
            flop_3_v = self.board[2].value
            flop_3_s = self.board[2].suit
            turn_v = self.board[3].value
            turn_s = self.board[3].suit
            river_v = -1
            river_s = -1
        else:
            flop_1_v = self.board[0].value
            flop_1_s = self.board[0].suit
            flop_2_v = self.board[1].value
            flop_2_s = self.board[1].suit
            flop_3_v = self.board[2].value
            flop_3_s = self.board[2].suit
            turn_v = self.board[3].value
            turn_s = self.board[3].suit
            river_v = self.board[4].value
            river_s = self.board[4].value
        observation = copy.deepcopy([position, starting_stack_hero, current_stack_hero, bet_street_hero, bet_total_hero,
         c_1_v, c_1_s, c_2_v, c_2_s, street, pot_size, starting_stack_vil, current_stack_vil,
         bet_street_vil, bet_total_vil, flop_1_v,flop_1_s, flop_2_v,flop_2_s, flop_3_v,flop_3_s,
         turn_v, turn_s, river_v, river_s])
        return observation









