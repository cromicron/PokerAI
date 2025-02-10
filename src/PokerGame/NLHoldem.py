import os.path
import random
import collections
from PokerGame.HandComperator import compare_hands
import copy
import numpy as np
import json
from datetime import datetime



value_dict = {10: 'T', 11: 'J', 12: 'Q', 13:'K', 14: 'A'}
suit_dict = {0:'c', 1:'d', 2:'h', 3:'s'}
deck_encode = [(rank, suit) for rank in range(2, 15) for suit in range(4)]
card_to_int = {
    card: i for i, card in enumerate(deck_encode)
}
int_to_card = {
    i: card for i, card in enumerate(deck_encode)
}
deck = [(rank, suit) for rank in range(2, 15) for suit in range(4)]


class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
        self.representation = str(value)+suit_dict[suit] if value < 10 else value_dict[value]+suit_dict[suit]
        self.int = card_to_int[(self.value, self.suit)]

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
        self.position = None
    def get_holecard_representation(self):
        return self.holecards[0].representation+self.holecards[1].representation

    @property
    def hole_int(self):
        return [card.int for card in self.holecards]

class Pot:
    def __init__(self, players, pot_size):
        self.players = players
        self.pot_size = pot_size

class SizeError(Exception):
    pass

class Game:
    def __init__(self, n_players=2, stacks=[], hand_history = False, save_hand_history = False):
        self.n_players = n_players
        self.deck = Deck()
        self.players = []
        self.max_bet = 0 #max_bet current round
        if len(stacks) != n_players and stacks != []:
            raise ArithmeticError("number of stacks must equal number of players")

        if stacks == []:
            self.starting_stacks = [200 for player in range(n_players)]
        else:
            self.starting_stacks = stacks
        if min(self.starting_stacks) < 4:
            raise ValueError("Starting Stack has to be at least 2")

        for i in range(n_players):
            new_player = Player(i, self.starting_stacks[i])
            self.players.append(new_player)
        self.positions = collections.deque(self.players)
        self.board = []
        self.pot = 0
        if hand_history:
            self.hand_history = HandHistory(save = save_hand_history)
        self.last_action = None

    def new_hand(self, random_seat=False, first_hand = False, reset_to_starting = True):
        self.last_action = None
        if hasattr(self, "hand_history"):
            self.hand_history.clear_hand_history()
        if random_seat:
            random.shuffle(self.positions)
        else:
            if not first_hand:
                self.positions.rotate(-1)
        self.left_in_hand = copy.copy(self.positions)
        for player in self.players:
            player.position = self.positions.index(player)
        if reset_to_starting:
            for player in self.players:
                player.stack = player.starting_stack
                player.bet = 0
        self.positions[0].stack -= 1
        self.positions[0].bet += 1
        self.positions[1].stack -= 2
        self.positions[1].bet += 2
        self.pot = 3
        self.deck.shuffle()
        for i in range(len(self.positions)):
            self.positions[i].holecards=self.deck[2*i:2*i+2]
        self.next = collections.deque(self.positions) #the next player to act is on top of the que
        self.next.reverse()

        self.next.rotate(2)
        self.roundabout = copy.copy(self.next)
        self.street = 0
        self.finished = False
        self.max_bet = 2
        self.added = 2 #current amount added to previous max_bet. Necessary to know minbet
        self.board= []

        if hasattr(self, "hand_history"):
            button_seat = 0 if self.n_players == 2 else self.n_players -1
            self.hand_history.start_new_hand(0, (1, 2), button_seat)
            for player in self.players:
                seat = self.positions.index(player)
                player_name = player.name
                stack = player.starting_stack
                self.hand_history.add_player(seat, player_name, stack)

            small_blind = self.positions[0].name
            big_blind = self.positions[1].name
            self.hand_history.set_blinds(small_blind, 1, big_blind, 2)

    def get_state(self, player):
        """Gives all necessary info of the current state for the player"""
        starting_stacks = [player.starting_stack for player in self.positions]
        in_play = [int(player in self.left_in_hand) for player in self.positions]
        stacks = [player.stack for player in self.positions]
        bets = [player.bet for player in self.positions]
        is_acting = self.acting_player == player

        state = {
            "n_players": self.n_players,
            "position": player.position,
            "starting_stack": player.starting_stack,
            "starting_stacks": starting_stacks,
            "in_play": in_play,
            "stack": player.stack,
            "stacks": stacks,
            "street": self.street,
            "bets": bets,
            "holecards": player.holecards,
            "board": self.board,
            "is_acting": is_acting,
            "legal_actions": self.get_legal_actions() if is_acting else None,
            "legal_betsize": self.get_legal_betsize() if is_acting else None,
            "bet": player.bet,
            "last_action": self.last_action,
        }
        return state

    @property
    def board_int(self):
        if self.street == 1:
            return self.flop_int, None, None
        elif self.street == 2:
            return self.flop_int, self.turn_int, None
        elif self.street == 3:
            return self.flop_int, self.turn_int, self.river_int
        else:
            raise NotImplementedError
    @property
    def flop_int(self):
        return [card.int for card in self.board[:3]]

    @property
    def turn_int(self):
        return self.board[3].int

    @property
    def river_int(self):
        return self.board[4].int

    def compare(self,players):
        #takes in players and returns relative hand strengths as list
        hands = [self.board + [cards for cards in player.holecards] for player in players]
        strengths = compare_hands(hands)
        return strengths

    def showdown(self, n_run = 1):
        #compares hands of remaining players and distributes chips accordingly
        #check for side pot possibilities: When a player is all in and has less chips than any of the remaining
        #players starting stacks

        # save stacks before showdown
        for player in self.players:
            player.stack_before_showdown = player.stack
        if hasattr(self, "hand_history"):
            for player in self.left_in_hand:
                self.hand_history.record_action(4, player.name, "shows", amount=player.get_holecard_representation())

        result = np.array(self.compare(self.left_in_hand))
        if len(self.left_in_hand)<3 or 0 not in [player.stack for player in self.left_in_hand]: #two players can't have a side pot
            winners = np.flatnonzero(result == np.max(result))
            for winner in winners:
                self.left_in_hand[winner].stack += self.pot/len(winners)
            # After all betting rounds and showdown are complete:
            if hasattr(self, "hand_history"):
                stack_changes = {player.name: player.stack-player.starting_stack for player in self.players}
                self.hand_history.set_summary(
                    board_cards=[card.representation for card in self.board],
                    main_pot={"Amount": self.pot/len(winners), "Winners": [self.left_in_hand[winner].name for winner in winners]},
                    stack_changes = stack_changes
                )
        #for side-pots

        else:
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

            #add all players who have equal or more than all in player to new pot
            previous_smallest = 0
            strengths = dict(zip(self.left_in_hand,result))

            while len(all_ins) > 0:
                current_smallest = all_ins[0].starting_stack
                pot = len(self.left_in_hand)*(current_smallest-previous_smallest)



                for i in range(len(bets_folded)):
                    to_add = min(bets_folded[i], current_smallest-previous_smallest)

                    pot += to_add
                    bets_folded[i] -= to_add
                previous_smallest = current_smallest

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

                for winner in winners:
                    winner.stack += pot.pot_size/len(winners)


    def get_legal_actions(self):
        if self.finished:
            return None
        next_player = self.next[-1]
        if next_player.bet == self.max_bet:
           return 1,2

        else:
            if next_player.stack > self.max_bet - next_player.stack:
                # check if all others are allin:
                n_with_chips = 0
                for player in self.left_in_hand:
                    if player.stack != 0:
                        n_with_chips += 1
                if n_with_chips > 1:
                    return 0,1,2

            return 0,1

    def get_legal_betsize(self):
        legal_actions = self.get_legal_actions()
        if 2 not in legal_actions:
            return None
        player = self.next[-1]
        if self.max_bet == 0:
            return min(2, player.stack)
        else:
            minbet = self.max_bet+self.added # must match and add at least the amount that the previous raiser added
            return min(minbet, player.stack+player.bet)


    @property
    def acting_player(self):
        return self.next[-1]


    def implement_action(self, player, action, amount=None):
        self.last_action = {"player": player, "action": action, "amount": amount}
        if player != self.next[-1]:
            raise ValueError("It's not the chosen player's turn to act")

        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            errorMsg = str(action) + " is an illegal action. Please choose " + str(legal_actions)
            raise ValueError(errorMsg)

        if amount != None and amount > player.stack+player.bet:
            raise SizeError("player doesn't have enough chips to make that bets/raise")
        if action == 0:
            if hasattr(self, "hand_history"):
                self.hand_history.record_action(self.street, player.name, "Fold")
            self.next.pop()
            self.roundabout.pop()
            self.left_in_hand.remove(player)
            if len(self.left_in_hand)==1:
                #if after a fold, only one player is left, the round is finished and that player who is left, receives
                #the pot
                if hasattr(self, "hand_history"):
                    stack_changes = {player.name: player.stack - player.starting_stack for player in self.players}
                    self.hand_history.set_summary(
                        board_cards=[card.representation for card in self.board],
                        main_pot={"Amount": self.pot, "Winners": [self.left_in_hand[0].name]},
                        stack_changes=stack_changes
                    )
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
                    if hasattr(self, "hand_history"):
                        self.hand_history.record_action(self.street, player.name, "Call", player.stack)
                    self.pot += player.stack
                    player.bet += player.stack
                    player.stack = 0
                else:
                    if hasattr(self, "hand_history"):
                        self.hand_history.record_action(self.street, player.name, "Call", to_call)
                    self.pot += to_call
                    player.bet += to_call
                    player.stack -= to_call
            #remove from roundabout if no chips left
            else:
                if hasattr(self, "hand_history"):
                    self.hand_history.record_action(self.street, player.name, "Check")
            if player.stack == 0:
                self.roundabout.remove(player)

        else: #player bets or raises. Everytime he does, he must spcify a bet/raise size.
            if amount == None:
                raise TypeError("Specify bet or raise size!")

            #check for legal raise size. Size must be matching the current raise/bet and adding equal ammount
            #if raised, but only if player has enough.
            if amount < self.get_legal_betsize():
                 raise SizeError("illegal betsize!")

            if hasattr(self, "hand_history"):
                action_name = "Bet" if self.max_bet == 0 else "Raise"
                self.hand_history.record_action(self.street, player.name, action_name, amount)
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
                            if hasattr(self, "hand_history"):
                                self.hand_history.record_street_cards(
                                    1,
                                    [card.representation for card in self.board]
                                )
                        elif self.street ==1:
                            self.board.extend(self.deck[2*self.n_players+3:2*self.n_players +5])
                            if hasattr(self, "hand_history"):
                                self.hand_history.record_street_cards(2, self.board[3].representation)
                        else:
                            self.board.extend(self.deck[2*self.n_players+4:2*self.n_players +5])
                            if hasattr(self, "hand_history"):
                                self.hand_history.record_street_cards(2, self.board[4].representation)
                        self.street = 3
                        self.showdown()
                        self.finished = True

                    else:
                        self.street += 1
                        if self.street == 1:
                            self.board.extend(self.deck[2*self.n_players:2*self.n_players +3])
                            if hasattr(self, "hand_history"):
                                self.hand_history.record_street_cards(
                                    1,
                                    [card.representation for card in self.board]
                                )
                        else:
                            self.board.append(self.deck[2*self.n_players +self.street+1])
                            if hasattr(self, "hand_history"):
                                self.hand_history.record_street_cards(self.street, self.board[-1].representation)
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




class HandHistory:
    def __init__(self, save = False):
        self.clear_hand_history()  # Initialize an empty hand history
        self.save = save
    def clear_hand_history(self):
        self.hand_history = {
            "HandNumber": None,
            "Stakes": None,
            "DateTime": None,
            "ButtonSeat": None,
            "Players": [],
            "Blinds": {},
            "Actions": {
                "PreFlop": [],
                "Flop": {"Cards": [], "Actions": []},
                "Turn": {"Card": None, "Actions": []},
                "River": {"Card": None, "Actions": []},
                "ShowDown": []
            },
            "Summary": {
                "Board": [],
                "WinningHand": [],
                "PotDistribution": [],
                "StackChanges": []
            }
        }

    def start_new_hand(self, hand_number, stakes, button_seat):
        self.hand_history["HandNumber"] = hand_number
        self.hand_history["DateTime"] = datetime.now().isoformat()
        self.hand_history["Stakes"] = stakes
        self.hand_history["ButtonSeat"] = button_seat

    def add_player(self, seat_number, player_name, stack):
        self.hand_history["Players"].append({"Seat": seat_number, "Player": player_name, "Stack": stack})

    def set_blinds(self, small_blind_player, small_blind_amount, big_blind_player, big_blind_amount):
        self.hand_history["Blinds"] = {
            "Small": {"Player": small_blind_player, "Amount": small_blind_amount},
            "Big": {"Player": big_blind_player, "Amount": big_blind_amount}
        }

    def record_action(self, street, player_name, action, amount=None):
        if street == 0:
            street_name = "PreFlop"
        elif street == 1:
            street_name = "Flop"
        elif street == 2:
            street_name = "Turn"
        elif street == 3:
            street_name = "River"
        else:
            street_name = "Showdown"
        action_record = {"Player": player_name, "Action": action}
        if amount is not None:
            action_record["Amount"] = amount

        # Add the action to the correct stage
        if street_name == "PreFlop":
            self.hand_history["Actions"]["PreFlop"].append(action_record)
        elif street_name == "Flop":
            self.hand_history["Actions"]["Flop"]["Actions"].append(action_record)
        elif street_name == "Turn":
            self.hand_history["Actions"]["Turn"]["Actions"].append(action_record)
        elif street_name == "River":
            self.hand_history["Actions"]["River"]["Actions"].append(action_record)
        elif street_name == "ShowDown":
            self.hand_history["Actions"]["ShowDown"].append(action_record)

    def record_street_cards(self, street, cards):
        if street == 1:
            street_name = "Flop"
        elif street == 2:
            street_name = "Turn"
        else:
            street_name = "River"
        if street_name == "Flop":
            self.hand_history["Actions"][street_name]["Cards"] = cards
        elif street_name in ["Turn", "River"]:
            self.hand_history["Actions"][street_name]["Card"] = cards  # Only one card for turn and river

    def save_hand_history(self, base_directory, base_file_name):
        # Check if the base directory exists, and if not, create it
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)  # This will create all directories in the path that don't exist

        # Get the current time and format it as specified
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")  # Fixed the format to be correct as yyyyMMddHHmmss
        file_name = f"{base_file_name}_{current_time}.json"
        full_path = os.path.join(base_directory, file_name)

        with open(full_path, 'w') as file:
            json.dump(self.hand_history, file, indent=4)

    def set_summary(self, board_cards, main_pot, stack_changes: dict, side_pots=None):
        self.hand_history["Summary"]["Board"] = board_cards
        self.hand_history["Summary"]["MainPot"] = main_pot
        self.hand_history["Summary"]["SidePots"] = side_pots if side_pots is not None else []
        self.hand_history["Summary"]["StackChanges"] = stack_changes
        if self.save:
            self.save_hand_history("HandHistory", "epic_hand.json")
