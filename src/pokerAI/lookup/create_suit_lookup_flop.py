from itertools import product
import pickle
def encode_suits(suits):
    suit_dict = {suits[0]: 0}
    suit_values = [0]
    suits_encoded = [0]
    for suit in suits[1:]:
        if suit not in suit_dict:
            new_suit =  max(suit_values) + 1
            suit_dict[suit] = new_suit
            suits_encoded.append(new_suit)
            suit_values.append(new_suit)
        else:
            suits_encoded.append(suit_dict[suit])
    return suits_encoded

all_possibilities = product(range(4), repeat=5)
lookup_suits_flop = {}
for comb in all_possibilities:
    suits_encoded = encode_suits(list(comb))
    lookup_suits_flop[comb] = suits_encoded
with open("flop_suits_lookup.pkl", "wb") as f:
    pickle.dump(lookup_suits_flop, f)

if __name__ == "__main__":
    hand = [(5,3), (3,2), (4,1), (6, 3), (6,0)]
    suits_encoded = encode_suits([card[1] for card in hand])
    print(suits_encoded)
