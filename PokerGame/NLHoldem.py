import random
import collections
from HandComperator import compare_hands

class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

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
        self.holecards = None
        self.bet = 0 #how much has player put into the pot in the current betting round


class Game:
    def __init__(self, n_players=2, stacks=[]):
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

    def new_hand(self, random_seat=False, first_hand = False):
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
        self.street = 0
        self.finished = False
        self.max_bet =1
    def compare(self,players):
        #takes in players and returns relative hand strengths as dictonary
        hands = [self.board + [cards for cards in player.holecards] for player in players]
        strengths =  compare_hands(hands)



    def get_legal_actions(self):
        if self.finished:
            return None
        next_player = self.next[-1]
        print("stack sb ", self.positions[0].stack, " stack_bb ", self.positions[1].stack)
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
        if action != legal_actions or ((amount != None) and amount > player.stack):
            errorMsg = action + " is an illegal action. Please choose " + legal_actions
            raise ValueError(errorMsg)
            return





