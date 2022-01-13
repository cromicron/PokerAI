#should be replaced with a time efficient lookuptable
from itertools import combinations as combs

values = list(range(2,15))
suits = [0, 1, 2, 3]
deck = []
for value in values:
    for suit in suits:
        deck.append((value, suit))
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

