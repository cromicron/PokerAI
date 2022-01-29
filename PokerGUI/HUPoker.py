import random
import numpy as np
import itertools
agent="test"
print(agent)

values = list(range(2,15))
suits = [0, 1, 2, 3]
deck = []
for value in values:
    for suit in suits:
        deck.append((value, suit))

def deal_preflop(deck,n_players):
    hands = []                
    for i in range(n_players):
        hand = [deck[i], deck[i+n_players]]
        hand.sort()
        hands.append(tuple(hand))
    return hands

def deal_flop(deck,n_players):
    return (deck[n_players*2+1:n_players*2+4])

def deal_turn(deck,n_players):
    return deck[n_players*2+5]

def deal_river(deck,n_players):
    return deck[n_players*2+7]


#a function which gives the strength of a hand.

def check_straightflush(hand):
    hand = hand[:]
    hand.sort(reverse = True)
    for start in range(0,3):
        left = 6 - start

        inarow = 1
        current = start
        nxt = start +1
        while inarow + left > 4 and inarow <5:
            if hand[nxt][0] == hand[current][0] or (hand[nxt][0]+1 == hand[current][0] and hand[current][1] != hand[nxt][1]): #if nxt hand is same value or next in line, but not same suit:

                nxt += 1
                left -= 1

            elif hand[current][0] != hand[nxt][0]+1:
                break
            elif hand[current][1] != hand[nxt][1]:
                break

            else:
                inarow += 1
                left -= 1
                current = nxt
                nxt = current + 1

        if inarow >= 5:

            return True, hand[start][0]


    if inarow <5:
        straightflush = False
        #check if Ace to 5 straightflush is in hand:

        for suit in [0,1, 2, 3]:
            if (14, suit) in hand and (2,suit) in hand and (3, suit) in hand and (4, suit) in hand and (5, suit) in hand:

                straightflush = True
                return True, 5

        if not straightflush:

            return False


def check_quads(hand):
    handfreq = {}
    for card in hand:
        if card[0] not in handfreq:
            handfreq[card[0]]=1
        else:
            handfreq[card[0]]+= 1

    for hand in handfreq:
        if handfreq[hand]==4:
            quad = hand
            #höchste Beikarte ermitteln
            del(handfreq[hand])
            kicker = 2
            for card in handfreq.keys():
                if card > kicker:
                    kicker = card
            return True, [quad, kicker]

    return False


def check_fullhouse(hand):
    handfreq = {}
    for card in hand:
        if card[0] not in handfreq:
            handfreq[card[0]]=1
        else:
            handfreq[card[0]]+= 1
    if 3 in handfreq.values():
            trips = 2
            for hand in handfreq.keys():
                if handfreq[hand]==3 and hand >trips:
                    trips = hand
            del(handfreq[trips])
            pair = 1
            for hand in handfreq.keys():
                if handfreq[hand] in (2,3) and hand >pair:
                    pair = hand
            if pair >1:
                return True, [trips, pair]
    return False

def check_flush(hand):
    #count suits
    spades = []
    hearts = []
    clubs = []
    diamonds = []
    for card in hand:
        if card[1] ==0:
            spades.append(card[0])
        elif card[1] == 1:
            hearts.append(card[0])
        elif card[1] == 2:
            clubs.append(card[0])
        else: 
            diamonds.append(card[0])


    if len(spades) >=5:
        suit = spades
    elif len(hearts) >= 5:
        suit = hearts
    elif len(clubs) >= 5:
        suit = clubs
    elif len(diamonds)>= 5:
        suit = diamonds
    else:
        return False

    suit.sort(reverse=True)
    return True, suit[0:5]


def check_straight(hand):
    hand = hand[:]
    hand.sort(reverse = True)
    for start in range(0,3):
        left = 6 - start
        inarow = 1
        current = start
        nxt = start +1
        while inarow + left > 4 and inarow <5:

            if hand[nxt][0] == hand[current][0]:

                nxt += 1
                left -= 1

            elif hand[current][0] != hand[nxt][0]+1:
                break

            else:
                inarow += 1
                current = nxt
                nxt = current + 1
                left -=1
        if inarow >= 5:
            return True, hand[start][0]

    if inarow <5:
        straight = False
        #check if Ace to 5 straight is in hand:
        values = []
        for card in hand:
            values.append(card[0])
            if 14 in values and 2 in values and 3 in values and 4 in values and 5 in values:

                straight = True
                return True, 5

        if not straight:
            return False

def check_trips(hand):
    handfreq = {}
    for card in hand:
        if card[0] not in handfreq:
            handfreq[card[0]]=1
        else:
            handfreq[card[0]]+= 1
    for value in handfreq:
        if handfreq[value]==3:
            trips = value
            del(handfreq[value])
            cards=[]
            cards.append(trips)
            kickers = list(handfreq.keys())
            kickers.sort(reverse=True)
            cards.extend(kickers[0:2])
            return True, cards
    return False

def check_twopair(hand):
    handfreq = {}
    for card in hand:
        if card[0] not in handfreq:
            handfreq[card[0]]=1
        else:
            handfreq[card[0]]+= 1

    pairs = []
    highcard = 2
    for value in handfreq: #if freq == 2 append to pairs, else check for highcard

        if handfreq[value]==2:
            pairs.append(value)
        else:
            if value > highcard:
                highcard = value

    if len(pairs)>=2:
        pairs.sort(reverse=True)
        if len(pairs)>2 and pairs[2]>highcard:
            return True, [pairs[0], pairs[1], pairs[2]]
        return True, [pairs[0], pairs[1], highcard]
    else:
        return False

def check_pair(hand): #if True return list with pair-card and four strongest non-pair values
    hand = hand[:]
    hand.sort(reverse=True)
    cards = []
    for i in range(6):
        for j in range(i+1,7):

            if hand[i][0] == hand[j][0]:
                pair = hand[i][0]
                del(hand[i], hand[i])
                cards.append(pair)
                for k in range(3):
                    cards.append(hand[k][0])                                   
                return (True, cards)
    return False


def strength(hand): #Using the check hand functions above. Starting with the strongest hand. If hand is True, the function above returns True and the meaningful information about the hand. If not, it creates a
    #TypeError, because only False is retured, so nothing can be written into the output variable. We use this to profit from the try, except statements. This might look weired, but by that we only have to
    #run the functions one to test for the hand and create the output.

    try:
        a, output = check_straightflush(hand)
        return 8, output

    except TypeError:
        try:
            a, output = check_quads(hand)
            return 7, output
        except TypeError:
            try:
                a, output = check_fullhouse(hand)
                return 6, output
            except TypeError:
                try:
                    a, output = check_flush(hand)        
                    return 5, output
                except TypeError:
                    try:
                        a, output = check_straight(hand)
                        return 4, output
                    except TypeError:
                        try:
                            a, output = check_trips(hand)
                            return 3, output
                        except TypeError:
                            try:
                                a, output = check_twopair(hand)
                                return 2, output
                            except TypeError:
                                try:
                                    a, output = check_pair(hand)
                                    return  1, output
                                except TypeError:
                                    #if hand is not pair, it's highcard. We return strength of 0 and a sorted list of cardvalues
                                    highcards = []
                                    hand = hand[:]
                                    hand.sort(reverse = True)
                                    for i in range(5):
                                        highcards.append(hand[i][0])
                                    return 0, highcards

def compare(hands): #takes in a dictonary of hands with position as keys (sb = 0) and returns dictonary of positions with rank (0 is strongest)
    rankings = hands.copy()
    for position in hands:
        rankings[position] = (strength(hands[position]))
    #create a sorted list of tuples with positions and strengths
    sorted_hands = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    #create a dictonary of positions and ranking. Equal strengths should get same ranking.
    ranking_dict = {sorted_hands[0][0]:0}
    for i in range(1, len(sorted_hands)):
        if sorted_hands[i][1] == sorted_hands[i-1][1]:
            ranking_dict[sorted_hands[i][0]]=i-1
        else:
            ranking_dict[sorted_hands[i][0]]=i
    return ranking_dict

class HUPoker:   
    def __init__(self, stack_hero, stack_villain,random_stack = False, minstack = 5, maxstack = 200, n_run =20, hero=agent, villain = agent): #we are going to train the AI by playing against itself. Later we are going to implement
        #a user interface, to play against. 
        self.starting_hero = stack_hero
        self.starting_villain = stack_villain
        self.n_run = n_run #we want to reward plays by the aproximate ev, so here we define how many times to board is run after a showdown prior to the river.
        self.hero = hero #the agents that interact with the environment. 
        self.villain = villain
        self.random_stack = random_stack
        self.minstack, self.maxstack  = minstack, maxstack


    def reset(self): #shuffles the deck, seats the two players and returns position, stack sizes, hand (only suited or unsuited, because actual stuits are not important preflop) 
        if self.random_stack:
            self.starting_hero = np.random.randint(self.minstack,self.maxstack) 
            self.starting_villain = np.random.randint(self.minstack,self.maxstack)
        #keep track of action counts
        self.count_round = -1

        self.position = np.random.randint(2)
        if self.position == 0:
            self.stack_sb = self.starting_hero - 0.5 
            self.stack_bb = self.starting_villain - 1
        else:
            self.stack_bb = self.starting_hero - 1 
            self.stack_sb = self.starting_villain - 0.5
        self.bet_sb = 0.5 #keeps track of the bet chips by player in current street
        self.bet_bb = 1
        self.left_bet_round = [1,0] #which players are left to bet in this round
        self.pot = self.bet_sb + self.bet_bb    
        self.street = 0 #0: preflop, 1:flop, 2: turn, 3: river

        random.shuffle(deck)
        self.board_current = []

        self.holecards = deal_preflop(deck,2)[self.position]
        self.holecards_villain = deal_preflop(deck, 2)[0] if self.position == 1 else deal_preflop(deck, 2)[1]
        self.value_low = self.holecards[0][0]
        self.value_high = self.holecards[1][0]
        if self.holecards[0][1] == self.holecards[1][1]:
            self.suited =1
        else:
            self.suited = 0


        self.action_villain = -1 #meaningless value

        #initiate hand history to -1
        self.hh =[[],[],[],[]]
        for street in range(4):
            for i in range(20):
                self.hh[street].append(-1)

        self.suit_low = -1
        self.suit_high = -2
        self.flop1val = -1 #meaningless values
        self.flop1suit = -1 #meaningless values
        self.flop2val = -1 #meaningless values
        self.flop2suit = -1 #meaningless values
        self.flop3val = -1 #meaningless values
        self.flop3suit = -1 #meaningless values
        self.turnval = -1
        self.turnsuit = -1
        self.riverval = -1
        self.riversuit = -1
        done = False

        #constructing observation for villain to chose action on.
        if self.position == 1:
            position = 0
            value_low = self.holecards_villain[0][0]
            suit_low = self.holecards_villain[0][1]
            value_high = self.holecards_villain[1][0]
            suit_high = self.holecards_villain[1][1]
            suited = 1 if suit_low == suit_high else 0
            if self.street == 0:
                suit_high, suit_low = -1, -1

            observation_villain = self.create_observation_villain()
            self.action_villain = self.villain.choose_action(observation_villain)

            if self.action_villain == 0:
                self.stack_bb += 1.5
                bet = 0
                done = True

            elif self.action_villain == 1:
                bet = 0.5

            elif self.action_villain >= 2 and self.action_villain <6:#minraise - at least complete plus the last raised bit
                bet = 1.5

            elif self.action_villain == 6 or self.action_villain == 7: #2.5 total bet villain
                bet = 2.5*self.bet_bb - self.bet_sb

            elif self.action_villain == 8 or self.action_villain == 9: #3 total bet villain
                bet = 3*self.bet_bb - self.bet_sb

            elif self.action_villain == 10: #4 total bet villain
                bet = 4*self.bet_bb - self.bet_sb

            else: #allin
                bet = self.stack_sb

            self.bet_sb += bet
            self.stack_sb -= bet
            self.pot += bet
            self.left_bet_round = [1]
            self.hh[0][0] = bet
            self.count_round = 0
        self.create_observation()

        return self.observation, done


    def create_observation(self): #creates the observation array. A state in poker consists of all actions made in the entire hand. So we have to save all actions by hero and villain.
        #the best way to do so is to keep track of all bet sizes. We need enough empty slots to keep track of all bets. A check gets the value 0.
            self.observation = np.asarray([self.position, self.stack_sb, self.stack_bb, self.street, self.value_low, self.value_high, self.suited,  
                                       self.pot, self.bet_sb, self.bet_bb, self.suit_low, self.suit_high, 
                                        self.hh[0][0], self.hh[0][1], self.hh[0][2], self.hh[0][3],self.hh[0][4], self.hh[0][5], self.hh[0][6],self.hh[0][7], self.hh[0][8], self.hh[0][9],
                                        self.hh[0][10], self.hh[0][11], self.hh[0][12], self.hh[0][13],self.hh[0][14], self.hh[0][15], self.hh[0][16],self.hh[0][17], self.hh[0][18], self.hh[0][19], 
                                        self.hh[1][0], self.hh[1][1], self.hh[1][2], self.hh[1][3],self.hh[1][4], self.hh[1][5], self.hh[1][6],self.hh[1][7], self.hh[1][8], self.hh[1][9],
                                        self.hh[1][10], self.hh[1][11], self.hh[1][12], self.hh[1][13],self.hh[1][14], self.hh[1][15], self.hh[1][16],self.hh[1][17], self.hh[1][18], self.hh[1][19], 
                                        self.hh[2][0], self.hh[2][1], self.hh[2][2], self.hh[2][3],self.hh[2][4], self.hh[2][5], self.hh[2][6],self.hh[2][7], self.hh[2][8], self.hh[2][9],
                                        self.hh[2][10], self.hh[2][11], self.hh[2][12], self.hh[2][13],self.hh[2][14], self.hh[2][15], self.hh[2][16],self.hh[2][17], self.hh[2][18], self.hh[2][19],
                                        self.hh[3][0], self.hh[3][1], self.hh[3][2], self.hh[3][3],self.hh[3][4], self.hh[3][5], self.hh[3][6],self.hh[3][7], self.hh[3][8], self.hh[3][9],
                                        self.hh[3][10], self.hh[3][11], self.hh[3][12], self.hh[3][13],self.hh[3][14], self.hh[3][15], self.hh[3][16],self.hh[3][17], self.hh[3][18], self.hh[3][19],
                                       self.flop1val, self.flop1suit, self.flop2val, self.flop2suit, self.flop3val, self.flop3suit, 
                                           self.turnval, self.turnsuit, self.riverval, self.riversuit]).astype(np.float32)

    def create_observation_villain(self): #this is the last action by hero. If two betting rounds in a row, set to -1
        position = 0 if self.position == 1 else 1
        value_low = self.holecards_villain[0][0]
        suit_low = self.holecards_villain[0][1] if self.street != 0 else -1
        value_high = self.holecards_villain[1][0]
        suit_high = self.holecards_villain[1][1] if self.street != 0 else -1
        suited = 1 if suit_low == suit_high else 0


        return np.asarray([position, self.stack_sb, self.stack_bb, self.street, value_low, value_high, suited,  
                                   self.pot, self.bet_sb, self.bet_bb, suit_low, suit_high,
                                   self.hh[0][0], self.hh[0][1], self.hh[0][2], self.hh[0][3],self.hh[0][4], self.hh[0][5], self.hh[0][6],self.hh[0][7], self.hh[0][8], self.hh[0][9],
                                    self.hh[0][10], self.hh[0][11], self.hh[0][12], self.hh[0][13],self.hh[0][14], self.hh[0][15], self.hh[0][16],self.hh[0][17], self.hh[0][18], self.hh[0][19], 
                                    self.hh[1][0], self.hh[1][1], self.hh[1][2], self.hh[1][3],self.hh[1][4], self.hh[1][5], self.hh[1][6],self.hh[1][7], self.hh[1][8], self.hh[1][9],
                                    self.hh[1][10], self.hh[1][11], self.hh[1][12], self.hh[1][13],self.hh[1][14], self.hh[1][15], self.hh[1][16],self.hh[1][17], self.hh[1][18], self.hh[1][19], 
                                    self.hh[2][0], self.hh[2][1], self.hh[2][2], self.hh[2][3],self.hh[2][4], self.hh[2][5], self.hh[2][6],self.hh[2][7], self.hh[2][8], self.hh[2][9],
                                    self.hh[2][10], self.hh[2][11], self.hh[2][12], self.hh[2][13],self.hh[2][14], self.hh[2][15], self.hh[2][16],self.hh[2][17], self.hh[2][18], self.hh[2][19],
                                    self.hh[3][0], self.hh[3][1], self.hh[3][2], self.hh[3][3],self.hh[3][4], self.hh[3][5], self.hh[3][6],self.hh[3][7], self.hh[3][8], self.hh[3][9],
                                    self.hh[3][10], self.hh[3][11], self.hh[3][12], self.hh[3][13],self.hh[3][14], self.hh[3][15], self.hh[3][16],self.hh[3][17], self.hh[3][18], self.hh[3][19],
                                    self.flop1val, self.flop1suit, self.flop2val, self.flop2suit, self.flop3val, self.flop3suit, 
                                    self.turnval, self.turnsuit, self.riverval, self.riversuit]).astype(np.float32)

    def implement_action(self, action, player = 'hero'): #implements either heros or villains action to the point where either the next player is to act or the hand is finished
        bet = 0
        self.count_round += 1
        done =False
        next_to_act = self.left_bet_round.pop() #last in list acts first. BB acts last preflop, which we allready tool care of above. Otherwise BB acts first.
        #transform action into legal option
        if action == 0:
            if self.bet_sb == self.bet_bb: #player cannot fold, if he can check
                action = 1

        elif action > 1: #player cannot raise, if he has less than pot - self.bet
            if (player == 'hero' and self.position == 0 and self.bet_bb - self.bet_sb >= self.stack_sb) or (
                player == 'hero' and self.position == 1 and self.bet_sb - self.bet_bb >= self.stack_bb) or(
                player == 'villain' and self.position == 1 and self.bet_bb - self.bet_sb >= self.stack_sb) or (
                player == 'villain' and self.position == 0 and self.bet_sb - self.bet_bb >= self.stack_bb):
                action = 1


        if action == 0: #player folds. Action is already transformed into 1, if player is first in.
            if next_to_act == 0: #hero was sb
                self.stack_bb += self.pot

            else: #player is bb
                self.stack_sb += self.pot
            self.left_bet_round = []
            done = True #round is finished. self.reset should be performed outside this function
            self.hh[self.street][self.count_round] = bet
            done=True
        elif action == 1: #player calls/checks
            if next_to_act == 0: #player is sb
                if self.bet_sb == self.bet_bb: #bets are ballanced, so player can check.

                    pass               

                else: #bets are unbalanced, so player calls
                    left_to_bet = self.bet_bb - self.bet_sb
                    if left_to_bet < self.stack_sb: #hero doesn't have to go allin to call
                        bet = self.bet_bb-self.bet_sb
                        self.bet_sb = self.bet_bb
                        self.stack_sb -= left_to_bet
                        self.pot += left_to_bet

                    else: #player goes allin
                        self.pot += self.stack_sb
                        self.bet_sb = self.bet_sb + self.stack_sb
                        bet = self.stack_sb
                        self.stack_sb = 0
                        #give bb chips higher than sb_stack back
                        bet_difference = self.bet_bb - self.bet_sb
                        self.stack_bb += bet_difference
                        self.pot -= bet_difference
                        #self.hh[self.street][self.count_round] = bet
                        self.showdown()
                        done = True


            else: #player is bb
                if self.bet_sb == self.bet_bb: #bets are ballanced, so player can check.
                    pass


                else: #bets are unbalanced, so player calls
                    left_to_bet = self.bet_sb - self.bet_bb
                    if left_to_bet <= self.stack_bb: #hero doesn't have to go allin to call
                        bet = self.bet_sb -self.bet_bb
                        self.bet_bb = self.bet_sb
                        self.stack_bb -= left_to_bet
                        self.pot += left_to_bet
                        self.left_bet_round = []


                    else: #plyer goes allin 
                        self.pot += self.stack_bb
                        self.bet_bb = self.bet_bb + self.stack_bb
                        bet = self.stack_bb
                        self.stack_bb = 0
                        #give bb chips higher than bb_stack back
                        bet_difference = self.bet_sb - self.bet_bb
                        self.stack_sb += bet_difference
                        self.pot -= bet_difference


                        #self.hh[self.street][self.count_round] = bet
                        self.showdown()
                        done = True


        else: #player raises/bets From here on we have to change the code.
            # if unbet the bet variable indicates how much the player will increase his current betsize.
            if next_to_act == 0 and self.bet_bb == 0 or next_to_act == 1 and self.bet_sb == 0:
                if action == 2:
                    bet = 1
                elif action == 3:
                    bet = self.pot//10.0
                    if bet < 1: #bet has to be at least 1BB
                        bet = 1
                elif action == 4:
                    bet = self.pot//4.0
                    if bet < 1: #bet has to be at least 1BB
                        bet = 1
                elif action ==5:
                    bet = self.pot//3.0
                    if bet < 1: #bet has to be at least 1BB
                        bet = 1

                elif action == 6:
                    bet = self.pot//2.0
                elif action ==7:
                    bet = 3*self.pot//4.0
                elif action == 8:
                    bet = self.pot
                elif action == 9:
                    bet = self.pot*13//10.0
                elif action == 10:
                    bet = self.pot*2

                else:
                    bet = self.stack_sb if self.position == 0 else self.stack_bb

            #if pot is bet:
            else:

                if action == 2 or action == 3:#minraise - at least complete plus the last raised bit
                    if next_to_act == 0:
                        #preflop first in
                        if self.street == 0 and self.bet_sb == 0.5:
                            bet = 1.5
                        else:
                            bet = 2*(self.bet_bb - self.bet_sb)
                    else:
                        bet = 2*(self.bet_sb - self.bet_bb)
                        if bet < 1:
                            bet = 1#at least a bet of 1
                elif action == 4 or action == 5: # 2 total bet villain
                    if next_to_act == 0:
                        bet = 2*self.bet_bb - self.bet_sb
                    else:
                        bet = 2*self.bet_sb - self.bet_bb
                elif action == 6 or action == 7: #2.5 total bet villain
                    if next_to_act == 0:
                        bet = 2.5*self.bet_bb - self.bet_sb
                    else:
                        bet = 2.5*self.bet_sb - self.bet_bb

                elif action == 8 or action == 9: #3 total bet villain
                    if next_to_act == 0:
                        bet = 3*self.bet_bb - self.bet_sb
                    else:
                        bet = 3*self.bet_sb - self.bet_bb
                elif action == 10: #4 total bet villain
                    if next_to_act == 0:
                        bet = 4*self.bet_bb - self.bet_sb
                    else:
                        bet = 4*self.bet_sb - self.bet_bb

                else: #allin
                    bet = self.stack_sb if next_to_act == 0 else self.stack_bb

            if (bet >= self.stack_sb and next_to_act == 0) or (bet >= self.stack_bb and next_to_act == 1): #allin if not enough chips left.
                bet = self.stack_sb if next_to_act == 0 else self.stack_bb



            if next_to_act == 0: #player is sb
                self.bet_sb += bet
                self.pot += bet
                self.stack_sb -= bet
                self.left_bet_round = [1] if self.stack_bb != 0 else [] #on an agressive move, his opponent is next to act. If player went all in, no further action.

            else:
                self.bet_bb += bet
                self.pot += bet
                self.stack_bb -= bet
                self.left_bet_round = [0] if self.stack_sb != 0 else []


        if len(self.left_bet_round) == 0:
            if not done:
                self.bet_sb = 0 #put bet in round back to 0
                self.bet_bb = 0

                if self.street == 3 or self.stack_sb == 0 or self.stack_bb == 0:
                    self.hh[self.street][self.count_round] = bet
                    self.showdown()
                    done = True
                    return done

                if self.street == 0:
                    self.hh[self.street][self.count_round] = bet
                    self.street = 1
                    self.count_round =-1

                    self.suit_low = self.holecards[0][1]
                    self.suit_high = self.holecards[1][1]
                    flop = deal_flop(deck,2)
                    self.board_current = flop
                    self.flop1val = flop[0][0]
                    self.flop1suit = flop[0][1]
                    self.flop2val = flop[1][0]
                    self.flop2suit = flop[1][1]
                    self.flop3val = flop[2][0]
                    self.flop3suit = flop[2][1]

                    self.left_bet_round = [0,1]

                    self.create_observation()
                    return done

                elif self.street ==1:
                    self.hh[self.street][self.count_round] = bet
                    self.street = 2
                    self.count_round = -1
                    turn = deal_turn(deck,2)
                    self.board_current.append(turn)

                    self.turnval = turn[0]
                    self.turnsuit = turn[1]

                    self.left_bet_round = [0,1]


                    self.create_observation()
                    return done

                else:
                    self.hh[self.street][self.count_round] = bet
                    self.street = 3
                    self.count_round = -1
                    river = deal_river(deck,2)
                    self.board_current.append(river)
                    self.riverval = river[0]
                    self.riversuit = river[1]

                    self.left_bet_round = [0,1]


                    self.create_observation()
                    return done

        self.hh[self.street][self.count_round] = bet
        return done       


    def step(self, action): #implements hero's action all up to the point where either the hand is over, or the next action is asked
        done = self.implement_action(action, 'hero')
        self.action_villain = -1 #if villain has no action after hero, set back to no action. Meaning that hero bets two times in a row
        if not done and self.left_bet_round != []:
            if self.left_bet_round[-1] != self.position: #villain is next to act.
                #constructing observation for villain to chose action on.            
                observation_villain = self.create_observation_villain()
                self.action_villain = self.villain.choose_action(observation_villain)  
                done = self.implement_action(self.action_villain,'villain')

                   #check if another villains action must be implemented.
                if not done and self.left_bet_round != []: #player left in round.
                    if self.left_bet_round[-1] != self.position: #next to act is not hero.
                    #check if we have to implement another villains action. 
                        observation_villain = self.create_observation_villain()
                        self.action_villain = self.villain.choose_action(observation_villain)            
                        done = self.implement_action(self.action_villain,'villain')
        self.create_observation()
        return done







    def showdown(self):

        if self.street == 0:
            cards_left = 5 #how many cards left to deal
        else:
            cards_left = 3 - self.street

        share_pot_sb, share_pot_bb = 0, 0
        for i in range(self.n_run):

            board = self.board_current[:]


            board_remaining = random.sample(deck[8:],cards_left) #we don't have to take into accounts the cards already dealt, because we start with the river card anyway.
            board.extend(board_remaining)
            #self.board_current.extend(board_remaining)


            hand_hero_total = list(self.holecards) + board
            hand_villain_total=list(self.holecards_villain) + board

            comparison = compare({0:hand_hero_total, 1:hand_villain_total})
            if comparison[0] < comparison[1]:

                if self.position == 0:                    
                    share_pot_sb += self.pot
                else:
                    share_pot_bb += self.pot
            elif comparison[0] == comparison[1]:

                share_pot_sb += self.pot/2
                share_pot_bb += self.pot/2

            else:

                if self.position == 0:
                    share_pot_bb += self.pot
                else:
                    share_pot_sb += self.pot


        self.stack_sb += share_pot_sb/self.n_run

        self.stack_bb += share_pot_bb/self.n_run

        self.create_observation()
        done = True
        #return done
