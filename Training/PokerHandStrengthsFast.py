from treys import Card, Evaluator
from PokerHandStrengths import compare

values = list(range(2,15))
suits = [0, 1, 2, 3]
deck = []
for value in values:
    for suit in suits:
        deck.append((value, suit))
        
lookupTreys = {}
for card in deck:
    if card[0] < 10:
        valueTreys = str(card[0])
    elif card[0] == 10:
        valueTreys = "T"
    elif card[0] == 11:
        valueTreys = "J"
    elif card[0] == 12:
        valueTreys = "Q"
    elif card[0] == 13:
        valueTreys = "K"
    else:
        valueTreys = "A"
    if card[1] == 0:
        suitTreys = "s"
    elif card[1] == 1:
        suitTreys = "h"
    elif card[1] == 2:
        suitTreys = "c"
    else:
        suitTreys = "d"
    
    lookupTreys[card] = Card.new(valueTreys+suitTreys)

def transformHand(hand):
    return [lookupTreys[hand[0]],lookupTreys[hand[1]],lookupTreys[hand[2]],lookupTreys[hand[3]],lookupTreys[hand[4]],lookupTreys[hand[5]],lookupTreys[hand[6]]]
evaluator = Evaluator()
def compare(hands):
    hand0 = transformHand(hands[0])
    hand1 = transformHand(hands[1])

    p0_score = evaluator.evaluate(hand0[2:], hand0[:2])
    p1_score = evaluator.evaluate(hand1[2:], hand1[:2])
    if p0_score < p1_score:
        return {0:0, 1:1}
    elif p1_score < p0_score:
        return {1:0, 0:1}
    else:
        return {0:0, 1:0}
        