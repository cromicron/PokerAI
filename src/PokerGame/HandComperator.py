def check_straightflush(hand):
    hand = hand[:]
    hand.sort(reverse=True)
    for start in range(0, 3):
        left = 6 - start

        inarow = 1
        current = start
        nxt = start + 1
        while inarow + left > 4 and inarow < 5:
            if hand[nxt][0] == hand[current][0] or (
                    hand[nxt][0] + 1 == hand[current][0] and hand[current][1] != hand[nxt][
                1]):  # if nxt hand is same value or next in line, but not same suit:

                nxt += 1
                left -= 1

            elif hand[current][0] != hand[nxt][0] + 1:
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

    if inarow < 5:
        straightflush = False
        # check if Ace to 5 straightflush is in hand:

        for suit in [0, 1, 2, 3]:
            if (14, suit) in hand and (2, suit) in hand and (3, suit) in hand and (4, suit) in hand and (
            5, suit) in hand:
                straightflush = True
                return True, 5

        if not straightflush:
            return False, 0


def check_quads(hand):
    handfreq = {}
    for card in hand:
        if card[0] not in handfreq:
            handfreq[card[0]] = 1
        else:
            handfreq[card[0]] += 1

    for hand in handfreq:
        if handfreq[hand] == 4:
            quad = hand
            # höchste Beikarte ermitteln
            del (handfreq[hand])
            kicker = 2
            for card in handfreq.keys():
                if card > kicker:
                    kicker = card
            return True, [quad, kicker]

    return False, 0


def check_fullhouse(hand):
    handfreq = {}
    for card in hand:
        if card[0] not in handfreq:
            handfreq[card[0]] = 1
        else:
            handfreq[card[0]] += 1
    if 3 in handfreq.values():
        trips = 2
        for hand in handfreq.keys():
            if handfreq[hand] == 3 and hand > trips:
                trips = hand
        del (handfreq[trips])
        pair = 1
        for hand in handfreq.keys():
            if handfreq[hand] in (2, 3) and hand > pair:
                pair = hand
        if pair > 1:
            return True, [trips, pair]
    return False, 0


def check_flush(hand):
    # count suits
    spades = []
    hearts = []
    clubs = []
    diamonds = []
    for card in hand:
        if card[1] == 0:
            spades.append(card[0])
        elif card[1] == 1:
            hearts.append(card[0])
        elif card[1] == 2:
            clubs.append(card[0])
        elif card[1] == 3:
            diamonds.append(card[0])

    if len(spades) >= 5:
        suit = spades
    elif len(hearts) >= 5:
        suit = hearts
    elif len(clubs) >= 5:
        suit = clubs
    elif len(diamonds) >= 5:
        suit = diamonds
    else:
        return False, 0

    suit.sort(reverse=True)
    return True, suit[0:5]


def check_straight(hand):
    hand = hand[:]
    hand.sort(reverse=True)
    for start in range(0, 3):
        left = 6 - start
        inarow = 1
        current = start
        nxt = start + 1
        while inarow + left > 4 and inarow < 5:

            if hand[nxt][0] == hand[current][0]:

                nxt += 1
                left -= 1

            elif hand[current][0] != hand[nxt][0] + 1:
                break

            else:
                inarow += 1
                current = nxt
                nxt = current + 1
                left -= 1
        if inarow >= 5:
            return True, hand[start][0]

    if inarow < 5:
        straight = False
        # check if Ace to 5 straight is in hand:
        values = []
        for card in hand:
            values.append(card[0])
            if 14 in values and 2 in values and 3 in values and 4 in values and 5 in values:
                straight = True
                return True, 5

        if not straight:
            return False, 0


def check_trips(hand):
    handfreq = {}
    for card in hand:
        if card[0] not in handfreq:
            handfreq[card[0]] = 1
        else:
            handfreq[card[0]] += 1
    for value in handfreq:
        if handfreq[value] == 3:
            trips = value
            del (handfreq[value])
            cards = []
            cards.append(trips)
            kickers = list(handfreq.keys())
            kickers.sort(reverse=True)
            cards.extend(kickers[0:2])
            return True, cards
    return False, 0


def check_twopair(hand):
    handfreq = {}
    for card in hand:
        if card[0] not in handfreq:
            handfreq[card[0]] = 1
        else:
            handfreq[card[0]] += 1

    pairs = []
    highcard = 2
    for value in handfreq:  # if freq == 2 append to pairs, else check for highcard

        if handfreq[value] == 2:
            pairs.append(value)
        else:
            if value > highcard:
                highcard = value

    if len(pairs) >= 2:
        pairs.sort(reverse=True)
        if len(pairs) > 2 and pairs[2] > highcard:
            return True, [pairs[0], pairs[1], pairs[2]]
        return True, [pairs[0], pairs[1], highcard]
    else:
        return False, 0


def check_pair(hand):  # if True return list with pair-card and four strongest non-pair values
    hand = hand[:]
    hand.sort(reverse=True)
    cards = []
    for i in range(6):
        for j in range(i + 1, 7):

            if hand[i][0] == hand[j][0]:
                pair = hand[i][0]
                del (hand[i], hand[i])
                cards.append(pair)
                for k in range(3):
                    cards.append(hand[k][0])
                return (True, cards)
    return False, 0


def strength(
        hand):  # One after the other we are going to check for hand strength. The return satatement will end the function and give back the hand strength type
    # and subcategory of strength, if the function that checks for the particular hand type returned true.

    isType, substrength = check_straightflush(hand)
    if isType:
        strength_val = 8 + 1e-2 * substrength
        return strength_val

    isType, substrength = check_quads(hand)
    if isType:
        strength_val = 7 + 1e-2 * substrength[0] + 1e-4 * substrength[1]
        return strength_val

    isType, substrength = check_fullhouse(hand)
    if isType:
        strength_val = 6 + 1e-2 * substrength[0] + 1e-4 * substrength[1]
        return strength_val

    isType, substrength = check_flush(hand)
    if isType:
        strength_val = 5 + 1e-2 * substrength[0] + 1e-4 * substrength[1] + 1e-6 * substrength[2] + 1e-8 * substrength[
            3] + 1e-10 * substrength[4]
        return strength_val

    isType, substrength = check_straight(hand)
    if isType:
        strength_val = 4 + 1e-2 * substrength
        return strength_val

    isType, substrength = check_trips(hand)
    if isType:
        strength_val = 3 + 1e-2 * substrength[0] + 1e-4 * substrength[1] + 1e-6 * substrength[2]
        return strength_val

    isType, substrength = check_twopair(hand)
    if isType:
        strength_val = 2 + 1e-2 * substrength[0] + 1e-4 * substrength[1] + 1e-6 * substrength[2]
        return strength_val

    isType, substrength = check_pair(hand)
    if isType:
        strength_val = 1 + 1e-2 * substrength[0] + 1e-4 * substrength[1] + 1e-6 * substrength[2] + 1e-8 * substrength[3]
        return strength_val

    # if hand is not pair, it's highcard. We return strength of 0
    highcards = []
    hand = hand[:]
    hand.sort(reverse=True)
    for i in range(5):
        highcards.append(hand[i][0])
    return 0 + 1e-2 * highcards[0] + 1e-4 * highcards[1] + 1e-6 * highcards[2] + 1e-8 * highcards[3] + 1e-10 * \
           highcards[4]


def compare_hands(hands: list):
    # Takes in lists of seven poker cards and returns the value of each hand
    strengths = [strength([(card.value, card.suit) for card in player]) for player in hands]
    return strengths
