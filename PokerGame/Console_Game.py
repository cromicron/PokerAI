from NLHoldem import Game

def print_stacks(all=False):
    if all:
        for i in range(game.n_players):
            print(positions_dict[i], ': ', game.positions[i].stack)
    else:
        for i in range(game.n_players):
            if game.positions[i] in game.left_in_hand:
                print(positions_dict[i], ': ', game.positions[i].stack)

def print_hands(all=False):
    print('holecards')
    if all:
        for i in range(game.n_players):
            print(positions_dict[i], ': ', game.positions[i].get_holecard_representation())

def print_board():
    board_str = ""
    for card in game.board:
        board_str+=card.representation
    print(board_str)
try:
    n_players = int(input("how many players?"))
except ValueError:
    print("Choose 2 - 9 players")
    n_players = None
while n_players not in list(range(2,10)):
    try:
        n_players = int(input("how many players?"))
    except ValueError:
        print("Choose between 2 and 9 players")

game = Game(n_players)
if game.n_players == 2:
    positions_dict = {0:'SB', 1:'BB'}
elif game.n_players == 3:
    positions_dict = {0:'SB', 1:'BB', 2:'BU'}
elif game.n_players == 4:
    positions_dict = {0:'SB', 1:'BB', 2:'CU', 3:'BU'}
elif game.n_players == 5:
    positions_dict = {0:'SB', 1:'BB', 2:'UTG', 3:'CU', 4:'BU'}
elif game.n_players == 6:
    positions_dict = {0:'SB', 1:'BB', 2: 'UTG', 3:'HJ', 4:'CU', 5:'BU'}
elif game.n_players == 7:
    positions_dict = {0:'SB', 1:'BB', 2: 'UTG', 3:'LJ', 4:'HJ', 5:'CU', 6:'BU'}
elif game.n_players == 8:
    positions_dict = {0:'SB', 1:'BB', 2: 'UTG', 3: 'UTG+1', 4:'LJ', 5:'HJ', 6:'CU', 7:'BU'}
else:
    positions_dict = {0:'SB', 1:'BB', 2:'UTG', 3: 'UTG+1', 4: 'UTG+2', 5:'LJ', 6:'HJ', 7:'CU', 8:'BU'}

game.new_hand(first_hand = True)
print("Stacks:")
print_stacks()
print_hands(True)

#start betting round preflop
#get possible actions
while not game.finished:
    street = game.street
    current_player = game.next[-1]
    next = game.positions.index(current_player)
    position_next = positions_dict[next]

    legal_actions = game.get_legal_actions()
    print("pot:",game.pot)
    actions_dict = {0: "fold", 1: "check/call", 2: "bet/raise"}
    act_str = " "
    for action in legal_actions:
        act_str += str(action)+":"+actions_dict[action]+" "
    print("choose action for",position_next+"("+current_player.get_holecard_representation()+").",act_str)
    try:
        action = int(input())
    except ValueError:
        print("choose a valid action")
    while action not in legal_actions:
        print("choose a valid action")
        try:
            action = int(input())
        except ValueError:
            print("choose a valid action")
    if action == 2:
        size = int(input("choose bet/raise-size\n"))
        while size < game.max_bet + game.added:
            size = int(input("illegal betsize. Choose bet/raise-size\n"))
    game.implement_action(current_player,action) if action !=2 else game.implement_action(current_player,action,size)
    print_stacks()
    if street != game.street:
        if game.street == 1:
            print("dealing flop")
            print_board()
        elif game.street == 2:
            print("dealing turn")
            print_board()
        else:
            print("dealing river")
            print_board()

    if game.finished:
        print("hand finished. Stack Sizes")
        print_stacks(all=True)



