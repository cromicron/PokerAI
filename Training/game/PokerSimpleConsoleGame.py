#console game
from PokerSimple import PokerSimple
import pickle
import numpy as np
with open('q_simple', 'rb') as q_simple_file:
    q = pickle.load(q_simple_file)
    
hero = 0
villain = 1
score_hero = 0
score_villain = 0

game = PokerSimple(hero, villain, False)
def new_hand():
    print("\n ---------------------------------- \n start new hand")
    global score_hero, score_villain
    done, next_to_act, observation = game.reset()
    hands = [game.hole_0,game.hole_1]

    sb = next_to_act
    bb = 0 if sb==1 else 1
    if sb == 0:
        print("You are sb")

    else:
        print("You are bb")
    print("Your hand ", game.hole_0)

    if sb == 1:
        action = np.argmax(q['sb'][hands[1]]['preflop']['preflop_fi'])
        if action == 0:
            print("Villain folds")
            game.implement_action(villain, 0)

        elif action == 1:
            game.implement_action(villain, 1)
            print("Villain calls")
            print("Pot is 2 BB")
            print("Your stack is 4 BB, Villains stack is 4 BB")
            try:
                action_hero = int(input("Choose action. 1: Check, 2: Allin"))
            except ValueError:
                action_hero = ""
            while action_hero not in [1,2]:
                try:
                    action_hero = int(input("Wrong Input. Choose action. 1: Check, 2: Allin"))
                except ValueError:
                    pass
                
            game.implement_action(hero, action_hero)
        else:
            game.implement_action(villain, 2)
            print("Villain goes all in")
            print("Pot is 6 BB")
            print("Your stack is 4 BB, Villains stack is 0 BB")
            try:
                action_hero = int(input("Choose action. 0: Fold, 1: Call Allin"))
            except ValueError:
                action_hero = ""
            while action_hero not in [0,1]:
                try:
                    action_hero = int(input("Wrong Input. Choose action. 0: Fold, 1: Call Allin"))
                except:
                    pass
            game.implement_action(hero, action_hero)

    done = game.done

    while not done:
        if game.next_to_act[0] == 0:
            print("The street is", game.street)
            if game.street == 1:
                print("The card on the board is ", game.board)
            print("It's your turn")
            print("Your stack is ", game.stacks[0])
            print("The pot is ", game.pot)

            if game.stacks[1] == 0:
                try:
                    action_hero = int(input("Choose action. 0: Fold, 1: Call Allin"))
                except ValueError:
                    pass
                
                while action_hero not in [0,1]:
                    try:                    
                        action_hero = int(input("Wrong Input. Choose action. 0: Fold, 1: Call Allin"))
                    except ValueError:
                        print("You must enter an integer")
                if action_hero ==1:
                    print("Showdown")
                    print("Villain shows ", game.hole_1)

            elif game.street == 0:
                try:
                    action_hero = int(input("Choose action. 0: Fold, 1: Call, 2: Allin"))
                except ValueError:
                    action_hero = ""
                while action_hero not in [0,1,2]:
                    try:
                        action_hero = int(input("Wrong Input. Choose action. 0: Fold, 1: Call, 2: Allin"))
                    except ValueError:
                        pass

            else:
                try:
                    action_hero =int(input("Choose action. 1: Check, 2: Allin"))
                except ValueError:
                    action_hero = ""
                while action_hero not in [1,2]:
                    try:
                        action_hero = int(input("Wrong Input. Choose action. 1: Check, 2: Allin"))
                    except ValueError:
                        pass
                if action_hero == 1 and game.next_to_act == [hero]:
                    print("Showdown")
                    print("Villain shows ", game.hole_1)
                    

            game.implement_action(hero, action_hero)
            done= game.done

        else:
            if game.next_to_act[0] == sb:
                if game.street == 0: #if game is preflop and player is sb, then it must be that bb pushed, otherwise there is no more action for sb              
                    action = np.argmax(q['sb'][hands[game.next_to_act[0]]]['preflop']['preflop_to_push'])

                else:
                    if game.stacks[bb] != 0: #bb checked
                        action = np.argmax(q['sb'][hands[game.next_to_act[0]]]['postflop'][game.board]['to_check'])

                        if action == 0:
                            action =1

                        else:
                            action = 2


                    else: ##bb pushed
                        action = np.argmax(q['sb'][hands[game.next_to_act[0]]]['postflop'][game.board]['to_push'])

            else:
                if game.street == 0:
                    if game.stacks[sb] != 0: #sb called                
                        action = np.argmax(q['bb'][hands[game.next_to_act[0]]]['preflop']['to_call'])

                        if action == 0:
                            action =1

                        else:
                            action = 2

                    else:
                        action = np.argmax(q['bb'][hands[game.next_to_act[0]]]['preflop']['to_push'])

                else:
                    game.create_observation(game.next_to_act[0])
                    if game.observations[game.next_to_act[0]][9] == -1: #postflop first in can only happen if preflop went call check           
                        action = np.argmax(q['bb'][hands[game.next_to_act[0]]]['postflop'][game.board]['postflop_fi'])

                        if action == 0:
                            action = 1

                        else:
                            action = 2

                    else: #if there is any action left it must be that smallblind pushed
                        action = np.argmax(q['bb'][hands[game.next_to_act[0]]]['postflop'][game.board]['postflop_to_push'])
                        last_state_action[game.next_to_act[0]]['state'] = ['bb',hands[game.next_to_act[0]], 'postflop', game.board, 'postflop_to_push']
                        last_state_action[game.next_to_act[0]]['action'] = action



            if action == 0:
                print("Villain folds.")
                print("You win ", game.pot, " BB.")
            elif action ==1:
                if game.stacks[0]==game.stacks[1]: #villain checks
                    print("Villain checks")
                    if game.next_to_act == [1]:
                        print("Showdown")
                        print("Villain shows ", game.hole_1)                    
                else:
                    print("Villain calls.")
                    print("Showdown")
                    print("Villain shows ", game.hole_1)
            else:
                print("Villain goes allin.")
            game.implement_action(villain,action)            
            done = game.done
    if done:
        score_hero += game.stacks[0]-5
        score_villain += game.stacks[1]-5
        if game.stacks[0]-5 > 0:
            print("You win ", game.pot, "BB.")
        elif game.stacks[0]-5 < 0:
            print("You loose. Villain wins ", game.pot, "BB.")
        else:
            print("Split Pot")
        
        print("Your score changed by ", game.stacks[0]-5, " BB.")
        print("your new score is ", score_hero)
        
        new_round = input("Do you want to play another hand? y or n")
        while new_round not in ['y', 'Y', 'n', 'N']:
            new_round = input("Wrong input. Do you want to play another hand? y or n")
        if new_round.lower() == 'y':
            new_hand()
new_hand()