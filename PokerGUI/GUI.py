#from ai import agent #ai to play against
from Agent import Agent
from Agent2 import Agent2
handsPlayed = -1
lr = 0.001
gamma = 1
n_actions = 12
epsilon = 0
batch_size = 32
input_dims = (102,)
stacksize = 10
if stacksize == 20:
    fname='PokerNewAi2.h5'
elif stacksize == 10:
    fname='PokerAI10BB.h5'
else:
    fname='PokerAI100.h5'
agent = Agent(lr, gamma, n_actions, epsilon, batch_size, input_dims, stacksize, fname = fname, handEval=True)
agent.load_model()

from gameplay import HUPoker
import random
from tkinter import *
from PIL import ImageTk, Image
import time
import threading
import numpy as np

root = Tk()
root.title("Poker")
#root.iconbitmap('icon.ico')
root.configure(background='green')
root.geometry("700x500")
root.grid_rowconfigure(0, minsize=80, weight=1)
root.grid_columnconfigure(0, minsize=60, weight=1)
#root.grid_columnconfigure(1, weight=1)

#creating a deck dictonary to transform cards from gameplay into images
values = list(range(2,15))
suits = [0, 1, 2, 3]
deck = []
for value in values:
       for suit in suits:
            deck.append((value, suit))
value_dict = {14: 'Ace of ', 13: 'King of ', 12: 'Queen of ', 11: 'Jack of ',
              10: 'Ten of ', 9: 'Nine of ', 8: 'Eight of ', 7: 'Seven of ',
              6: 'Six of ', 5: 'Five of ', 4: 'Four of ', 3: 'Three of ',
              2: 'Two of '}
suit_dict = {0: 'Clubs.png', 1: 'Diamonds.png', 2: 'Spades.png', 3: 'Hearts.png'}
card_dict = {}
for card in deck:
    
    card_dict[card] = value_dict[card[0]]+suit_dict[card[1]]


#define frames
frame_main=Frame(root, bg='green')
frame_main.grid()
frame_hero = Frame(frame_main,bg='green')
frame_villain = Frame(frame_main, bg='green')
frame_center = Frame(frame_main, bg='green',pady=20)
frame_action = Frame(frame_main, bg='green', padx=10, pady=25)
frame_score = Frame(frame_main, bg='green')

#position frames
frame_hero.grid (column=0, row = 2, sticky = 'nsew')
frame_villain.grid(column=0,row =0, sticky = 'nsew')
frame_center.grid (column=0,row = 1,  sticky = 'nsew')
frame_action.grid(column=1,row = 2,  sticky = 'nsew')
frame_score.grid(column=2, row=0,  sticky = 'nsew')

frame_board= Frame(frame_center, bg='green')
#frame_board.grid_rowconfigure(0,minsize = 100)
frame_board.grid(row = 0)
#images
size_cards = (50,70)
#initiate game
game = HUPoker(stacksize,stacksize)
game.reset()
c1_hero = Image.open('cards/'+ card_dict[game.holecards_villain[0]])
c2_hero = Image.open('cards/'+ card_dict[game.holecards_villain[1]])
c1_hero= c1_hero.resize(size_cards, Image.ANTIALIAS)
c2_hero= c2_hero.resize(size_cards, Image.ANTIALIAS)
c1_hero = ImageTk.PhotoImage(c1_hero)
c2_hero = ImageTk.PhotoImage(c2_hero)
c1_hero_label = Label(frame_hero, image=c1_hero, bg='green')
c2_hero_label = Label(frame_hero,image=c2_hero, bg='green')

def update_hand_img(fold = False):
    if not fold:
        c1_hero = Image.open('cards/'+ card_dict[game.holecards_villain[0]])
        c2_hero = Image.open('cards/'+ card_dict[game.holecards_villain[1]])
        c1_hero= c1_hero.resize(size_cards, Image.ANTIALIAS)
        c2_hero= c2_hero.resize(size_cards, Image.ANTIALIAS)
        c1_hero = ImageTk.PhotoImage(c1_hero)
        c2_hero = ImageTk.PhotoImage(c2_hero)
        c1_hero_label.configure(image=c1_hero)
        c2_hero_label.configure(image=c2_hero)
        c1_hero_label.image = c1_hero
        c2_hero_label.image = c2_hero
    else:
        c1_hero_label.configure(image="")
        c2_hero_label.configure(image="")

    

back = Image.open('cards/card_back.jpg')
c1_villain = Image.open('cards/card_back.jpg')
c2_villain = Image.open('cards/card_back.jpg')
c1_flop = Image.open('cards/card_back.jpg')
c2_flop = Image.open('cards/card_back.jpg')
c3_flop = Image.open('cards/card_back.jpg')
c_turn = Image.open('cards/card_back.jpg')
c_river = Image.open('cards/card_back.jpg')
d_button = Image.open('button.png')


    
back = back.resize(size_cards, Image.ANTIALIAS)
c1_villain= c1_villain.resize(size_cards, Image.ANTIALIAS)
c2_villain= c2_villain.resize(size_cards, Image.ANTIALIAS)
c1_flop = c1_flop.resize(size_cards, Image.ANTIALIAS)
c2_flop = c2_flop.resize(size_cards, Image.ANTIALIAS)
c3_flop =c3_flop.resize(size_cards, Image.ANTIALIAS)
c_turn = c_turn.resize(size_cards, Image.ANTIALIAS)
c_river =c_turn.resize(size_cards, Image.ANTIALIAS)
d_button = d_button.resize((20,20), Image.ANTIALIAS)


back = ImageTk.PhotoImage(back)
c1_villain = ImageTk.PhotoImage(c1_villain)
c2_villain = ImageTk.PhotoImage(c2_villain)
d_button = ImageTk.PhotoImage(d_button)
d_button_hero = Label(frame_hero, image="", bg='green')
d_button_villain=Label(frame_villain, image="", bg='green')
d_button_hero.grid(row=3)
d_button_villain.grid(row=3)
def enter_act(event):
    pass

#to type in bet and raisesizes
def legal_input(char): #allow only numbers and dots in the entry field
    if char.isdigit() or char == ".":
        return True    
    else:
        return False
validation = frame_action.register(legal_input)
e = Entry(frame_action, width=5, validate = "key", validatecommand = (validation, '%S'))
e.bind('<Return>', enter_act)
e.insert(END,2)
e.grid(row=1, column =0, columnspan = 1)
def set_minbet():
    e.delete(0, END)
    if game.bet_sb == 0 and game.bet_bb == 0: #unbet so et e to 1
        minbet = 1
    elif game.position == 1:
        minbet =  2*(game.bet_bb - game.bet_sb)+game.bet_sb
    else:
        minbet = 2*(game.bet_sb -game.bet_bb)+game.bet_bb
    if minbet < 1:
        minbet = 1
    minbet = round(minbet,1)
    e.insert(0, str(minbet))


flop1 = Label(frame_board, image = "", bg='green')
flop1.grid(row=0, column=0)
flop2 = Label(frame_board, image = "", bg='green')
flop2.grid(row=0, column=1)
flop3 = Label(frame_board, image = "", bg='green')
flop3.grid(row=0, column=2)
turn = Label(frame_board, image = "", bg = 'green')
turn.grid(row = 0, column = 3)
river = Label(frame_board, image = "", bg = 'green')
river.grid(row = 0, column = 4)

label_pot = Label(frame_center, text = "", fg= 'white', background = 'green')
label_pot.grid(row = 1)

def update_flop_img():
    c1_flop = Image.open('cards/'+ card_dict[game.board_current[0]])
    c2_flop = Image.open('cards/'+ card_dict[game.board_current[1]])
    c3_flop = Image.open('cards/'+ card_dict[game.board_current[2]])
    c1_flop= c1_flop.resize(size_cards, Image.ANTIALIAS)
    c2_flop= c2_flop.resize(size_cards, Image.ANTIALIAS)
    c3_flop= c3_flop.resize(size_cards, Image.ANTIALIAS)
    c1_flop = ImageTk.PhotoImage(c1_flop)
    c2_flop = ImageTk.PhotoImage(c2_flop)
    c3_flop = ImageTk.PhotoImage(c3_flop)
    flop1.configure(image=c1_flop)
    flop2.configure(image=c2_flop)
    flop3.configure(image=c3_flop)
    flop1.image = c1_flop
    flop2.image = c2_flop
    flop3.image = c3_flop
    
def update_turn_img():
    c_turn = Image.open('cards/' + card_dict[game.board_current[3]])
    c_turn= c_turn.resize(size_cards, Image.ANTIALIAS)
    c_turn = ImageTk.PhotoImage(c_turn)
    turn.configure(image = c_turn)
    turn.image= c_turn
    
def update_river_img():
    c_river = Image.open('cards/' + card_dict[game.board_current[4]])
    c_river= c_river.resize(size_cards, Image.ANTIALIAS)
    c_river = ImageTk.PhotoImage(c_river)
    river.configure(image = c_river)
    river.image= c_river

def showdown_villain():
    c1_villain = Image.open('cards/'+ card_dict[game.holecards[0]])
    c2_villain = Image.open('cards/'+ card_dict[game.holecards[1]])
    c1_villain= c1_villain.resize(size_cards, Image.ANTIALIAS)
    c2_villain= c2_villain.resize(size_cards, Image.ANTIALIAS)
    c1_villain = ImageTk.PhotoImage(c1_villain)
    c2_villain = ImageTk.PhotoImage(c2_villain)
    c1_villain_label.configure(image=c1_villain)
    c2_villain_label.configure(image=c2_villain)
    c1_villain.image = c1_villain
    c2_villain.image = c2_villain
    
def step_player(action, betsize=0):
    global bet_hero, bet_villain, villain_chips,hero_chips    
    if action < 2:
        done = game.implement_action(action, 'villain')
    else:
        done = game.implement_action_player(betsize)
    hero_chips = game.stack_sb if game.position == 1 else game.stack_bb
    if not done and game.left_bet_round != []:
        if game.left_bet_round[-1] == game.position:
            game.create_observation()
            action_ai = agent.choose_action(game.observation)
            time.sleep(1)
            if action_ai == 0:
                bet_hero, bet_villain = "", ""
                c1_villain_label.configure(image="")
                c2_villain_label.configure(image="")
                labelvillain_action.configure(text ="FOLD!")
                
            elif action_ai == 1:
                if game.bet_sb == game.bet_bb:
                    labelvillain_action.configure(text ="CHECK!")
                else:
                    labelvillain_action.configure(text ="CALL!")
            else:
                if game.bet_sb == game.bet_bb:
                    labelvillain_action.configure(text ="RAISE!")
                else:
                    labelvillain_action.configure(text ="RAISE!")
            root.update()
            street = game.street
            done = game.implement_action(action_ai, 'hero')
            villain_chips = game.stack_sb if game.position == 0 else game.stack_bb
            
            if street != game.street:
                bet_hero, bet_villain = "", ""
                
            else:
                bet_villain = game.bet_sb if game.position == 0 else game.bet_bb
                labelvillain.configure(text="villain" + str(game.stack_sb) if game.position == 0 else "villain"+str(game.stack_bb))
                labelvillainbet.configure(text = bet_villain)
           
            root.update()
            
            if not done and game.left_bet_round != []:
                if game.left_bet_round[-1] == game.position:
                    game.create_observation()
                    action_ai = agent.choose_action(game.observation)
                    done = game.implement_action(action_ai, 'hero')
                    villain_chips = game.stack_sb if game.position == 0 else game.stack_bb
    if betsize > 0:
        if action_ai == 0: # if bet/raise player, we must know if ai folded
            return done, True
        else:
            return done, False
    else:
        return done

             
c1_villain_label = Label(frame_villain, image = back, bg='green')
c2_villain_label = Label(frame_villain, image = back, bg='green')
c1_villain_label.grid(column = 0, row=0)
c2_villain_label.grid(column=1, row = 0)
#c1_hero_label = Label(frame_hero, image=c1_hero, bg='green')
#c2_hero_label = Label(frame_hero,image=c2_hero, bg='green')
c1_hero_label .grid(column=0, row =1,)
c2_hero_label .grid(column=2, row = 1)


labelhero = Label(frame_hero, text = "hero", bg='green')
labelvillain = Label(frame_villain, text = "villain", bg='green')
labelvillain_action = Label(frame_villain, text="", bg='green')

labelherobet = Label(frame_hero, text ="", bg='green')
labelvillainbet = Label(frame_villain, text = "", bg='green')

labelhero.grid(row = 2)
labelvillain.grid(row = 1)
labelherobet.grid(row=0)
labelvillainbet.grid(row=2)
labelvillain_action.grid(row = 2, column = 1)



heroframe = Frame(frame_main, width = 4, bg='green')
heroframe.grid (row = 3)
score_hero = 0
score_ai = 0
label_score = Label(frame_score, text = "Hero " + str(score_hero)+ "\nVillain "+ str(score_ai),  anchor='w', justify= "left", bg='green')
label_score.grid(column=0, row = 0)
label_hands_played = Label(frame_score, text = str(handsPlayed), bg='green')
label_hands_played.grid(column = 1, row = 0)

def new_hand(game=game):
    global score_hero, score_ai, pot, bet_hero, bet_villain, hero_chips, villain_chips, handsPlayed
    handsPlayed += 1
    pot=0
    bet_hero = 0
    bet_villain = 0
    observation, done = game.reset()
    c1_villain_label.configure(image=back)
    c2_villain_label.configure(image=back)
    c1_villain_label.image=back
    c2_villain_label.image=back
    flop1.configure(image="")
    flop2.configure(image="")
    flop3.configure(image="")
    turn.configure(image="")
    river.configure(image="")
    labelvillain_action.configure(text ="")
    update_hand_img()
    label_pot.configure(text="")
    e.configure(state = NORMAL, fg='black')
    e.delete(0, END)
    e.insert(0, "2")
    
        #if villain is sb, AI acts.
    if game.position == 0:
        labelhero.configure(text = "hero "+str(game.starting_villain-1))
        labelvillainbet.configure(text = 0.5)
        action = agent.choose_action(observation)
        done = game.implement_action(action,'hero')
        pos_player=['hero','villain'] # reverse - hero for the ai
        d_button_villain.configure(image=d_button)
        d_button_villain.image=d_button
        d_button_hero.configure(image='')    
        print('ai is sb, action ', action)
        
    else:

        labelherobet.configure(text = 0.5)
        labelvillainbet.configure(text = 1)
        pos_player=['villain','hero']
        print('hero is sb')
        d_button_hero.configure(image=d_button)
        d_button_hero.image=d_button
        d_button_villain.configure(image='')
        labelvillain_action.configure(text ="")

    hero_chips = game.stack_sb if game.position == 1 else game.stack_bb
    villain_chips = game.stack_sb if game.position == 0 else game.stack_bb          
    labelhero.configure(text = "hero "+str(hero_chips))
    labelvillain.configure(text = "villain "+str(villain_chips))
    label_hands_played.configure(text = str(handsPlayed))
    root.update()
    
    #implement ai action if ai is sb
    if game.position == 0 and action == 0:
        bet_hero, bet_villain = "", ""
        c1_villain_label.configure(image="")
        c2_villain_label.configure(image="")
        labelvillain_action.configure(text ="FOLD!")
        pot = game.bet_sb + game.bet_bb
        label_pot.configure(text=pot)
        labelherobet.configure(text = "")
        labelvillainbet.configure(text = "")
        score_hero += 0.5
        score_ai -= 0.5
        label_score.configure(text = "Hero " + str(score_hero)+ "\nVillain "+ str(score_ai))
        root.update()
        time.sleep(1)

    elif game.position == 0 and action == 1:
        labelvillain_action.configure(text ="CALL!")
    elif game.position == 0 and action > 1:
        labelvillain_action.configure(text ="RAISE!")
    pot = 0
    
    bet_hero = 0.5 if game.position == 1 else 1
    bet_villain = game.starting_hero - game.stack_sb if game.position == 0 else game.starting_hero - game.stack_bb
    labelvillainbet.configure(text = bet_villain)
    labelvillain.configure(text = "villain "+str(villain_chips))
    
    labelherobet.configure(text = bet_hero)
    labelhero.configure(text = "hero "+str(hero_chips))    

    print(game.holecards_villain, c1_hero)
    if done:
        fold_button.configure(state=DISABLED, fg='grey')
        cc_button.configure(state=DISABLED, fg='grey')
        decideButton.configure(state=DISABLED, fg='grey')
        root.update()

        done, hero_chips, villain_chips, pot, bet_hero, bet_villain= new_hand()
    fold_button.configure(state=ACTIVE, fg='black')
    if bet_hero != bet_villain:
        cc_button.configure(state=ACTIVE,text= "Call", fg='black')
    else:
        cc_button.configure(state=ACTIVE,text= "Check", fg='black')
    
    decideButton.configure(state=ACTIVE, text = "Raise", fg='black')
    root.update()
    return done, hero_chips, villain_chips, pot, bet_hero, bet_villain

def button_betraise():
    global hero_chips, bet_hero, bet_villain, villain_chips, pot, score_hero, score_ai, pot
    fold_button.configure(state=DISABLED, fg='grey')
    cc_button.configure(state=DISABLED, fg='grey')
    decideButton.configure(state=DISABLED, fg='grey')
    e.configure(state=DISABLED, fg='grey')
    labelvillain_action.configure(text ="")
    bet_villain = game.bet_sb if game.position == 0 else game.bet_bb
    bet_hero = game.bet_sb if game.position == 1 else game.bet_bb
    street = game.street
    size = float(e.get())
    if size > (game.stack_sb + game.bet_sb ) and game.position == 1:
        size = game.stack_sb + game.bet_sb
    elif size > (game.stack_bb + game.bet_bb ) and game.position == 0:
        size = game.stack_bb + game.bet_bb

    print("bet hero ", bet_hero)
    hero_chips -= size - bet_hero
    bet_hero = size
    labelherobet.configure(text = bet_hero)
    labelhero.configure(text = "hero "+str(hero_chips))    
    root.update()

    done, ai_fold = step_player(2, size)
    if done: #show villain hand
        if not ai_fold:
            showdown_villain()
            root.update()
            time.sleep(1)
            if street < 1:
                update_flop_img()
                root.update()
                time.sleep(1)
            if street <2:
                update_turn_img()
                root.update()
                time.sleep(1)
            if street <3:
                update_river_img()
                root.update()
                time.sleep(1)
        if (game.starting_hero < game.stack_sb and game.position == 0) or  (game.starting_hero < game.stack_bb and game.position == 1):
            labelvillain_action.configure(text ="AI wins")
        elif (game.starting_hero == game.stack_sb and game.position == 0) or  (game.starting_hero == game.stack_bb and game.position == 1):
             labelvillain_action.configure(text ="Split pot")
        else:
             labelvillain_action.configure(text ="Hero wins")
        score_ai += game.stack_sb - game.starting_hero if game.position == 0 else game.stack_bb -game.starting_hero
        score_hero += game.stack_bb - game.starting_villain if game.position == 0 else game.stack_sb -game.starting_villain
        label_score.configure(text = "Hero " + str(score_hero)+ "\nVillain "+ str(score_ai))
        root.update()
        time.sleep(2)
        done, hero_chips, villain_chips, pot, bet_hero, bet_villain= new_hand()
        return
    root.update()
    
    if street != game.street:        
        
        labelvillain.configure(text = "ai "+str(villain_chips))
        labelhero.configure(text = "hero "+str(hero_chips))  
        labelvillainbet.configure(text = "")
        labelherobet.configure(text = "")
        bet_hero = 0
        bet_villain = game.bet_sb if game.position == 0 else game.bet_bb
        if game.street == 1:
            update_flop_img()
            pot = 1.5
            for bet in game.hh[0]:
                if bet == -1:
                    break
                else:
                    pot += bet
            label_pot.configure(text = pot)
            street = game.street
        elif game.street == 2:
            print("street: ", street, " game.street: ", game.street)
            update_turn_img()
            for bet in game.hh[1]:
                if bet == -1:
                    break
                else:
                    pot += bet
            label_pot.configure(text = pot)
            street = game.street
        else:
            update_river_img()
            for bet in game.hh[2]:
                if bet == -1:
                    break
                else:
                    pot += bet
            label_pot.configure(text = pot)
            street = game.street
        root.update()
        if bet_villain == 0:
            labelvillainbet.configure(text = "")
        else:
            labelvillainbet.configure(text = bet_villain)
        if bet_villain == 0 and bet_hero==0:
            decideButton.configure(state=ACTIVE,text= "Bet", fg='black')
        else:
            decideButton.configure(state=ACTIVE,text= "Raise", fg='black')
        time.sleep(1)
        root.update()
    
    if bet_hero != bet_villain:
        fold_button.configure(state=ACTIVE, fg='black')
        cc_button.configure(state=ACTIVE, text= "Call",fg='black')
    else:
        cc_button.configure(state=ACTIVE, text= "Check",fg='black')
    if bet_hero == 0 and bet_villain == 0:
        decideButton.configure(state=ACTIVE, text ="Bet", fg='black')     
    else:
        decideButton.configure(state=ACTIVE, text ="Raise", fg='black')
    e.configure(state=NORMAL, fg='black')
    set_minbet()
    root.update
    return done


def button_fold():
    global score_ai, score_hero, pot, bet_hero, bet_villain
    fold_button.configure(state=DISABLED, fg='grey')
    cc_button.configure(state=DISABLED, fg='grey')
    decideButton.configure(state=DISABLED, fg='grey')
    e.configure(state=DISABLED, fg='grey')
    root.update()
    done = game.implement_action(0, 'villain')
    score_ai += game.stack_sb - game.starting_hero if game.position == 0 else game.stack_bb -game.starting_hero
    score_hero += game.stack_bb - game.starting_villain if game.position == 0 else game.stack_sb -game.starting_villain
    label_score.configure(text = "Hero " + str(score_hero)+ "\nVillain "+ str(score_ai))
    update_hand_img(True)
    pot =pot + bet_hero
    label_pot.configure(text=pot)
    labelherobet.configure(text = "")
    labelvillainbet.configure(text = "")
    
    root.update()
    time.sleep(1)
    done, hero_chips, villain_chips, pot, bet_hero, bet_villain= new_hand()
    return done

def button_cc():
    global hero_chips, bet_hero, bet_villain, villain_chips, pot, score_hero, score_ai, pot
    fold_button.configure(state=DISABLED, fg='grey')
    cc_button.configure(state=DISABLED, fg='grey')
    decideButton.configure(state=DISABLED, fg='grey')
    e.configure(state=DISABLED, fg='grey')
    root.update()
    print("bet_hero ", bet_hero)
    print("bet_villain ", bet_villain)
    print("hero chips ", hero_chips)
    print("villain chips ", villain_chips)
    labelvillain_action.configure(text ="")
    if game.bet_sb == game.bet_bb: #check
        pass
    else: #call from sb
        bet_villain = game.bet_sb if game.position == 0 else game.bet_bb
        hero_chips -= (bet_villain-bet_hero)
        print("hero chips ",hero_chips)
        bet_hero = bet_villain        
        labelherobet.configure(text = bet_hero)
        labelhero.configure(text = "hero "+str(hero_chips))    
        root.update()
        
    street = game.street
    done = step_player(1)
    print(done)
    if done: #show villain hand
        showdown_villain()
        root.update()
        time.sleep(1)
        if street < 1:
            update_flop_img()
            root.update()
            time.sleep(1)
        if street <2:
            update_turn_img()
            root.update()
            time.sleep(1)
        if street <3:
            update_river_img()
            root.update()
            time.sleep(1)
        if (game.starting_hero < game.stack_sb and game.position == 0) or  (game.starting_hero < game.stack_bb and game.position == 1):
            labelvillain_action.configure(text ="AI wins")
        elif (game.starting_hero == game.stack_sb and game.position == 0) or  (game.starting_hero == game.stack_bb and game.position == 1):
             labelvillain_action.configure(text ="Split pot")
        else:
             labelvillain_action.configure(text ="Hero wins")
        score_ai += game.stack_sb - game.starting_hero if game.position == 0 else game.stack_bb -game.starting_hero
        score_hero += game.stack_bb - game.starting_villain if game.position == 0 else game.stack_sb -game.starting_villain
        label_score.configure(text = "Hero " + str(score_hero)+ "\nVillain "+ str(score_ai))
        root.update()
        time.sleep(2)
        done, hero_chips, villain_chips, pot, bet_hero, bet_villain= new_hand()
        return
        
                
    print(game.board_current)
    if street != game.street:
        
        
        labelvillain.configure(text = "ai "+str(villain_chips))
        labelhero.configure(text = "hero "+str(hero_chips))  
        labelvillainbet.configure(text = "")
        labelherobet.configure(text = "")
        bet_hero = 0
        bet_villain = game.bet_sb if game.position == 0 else game.bet_bb
        if game.street == 1:
            update_flop_img()
            pot = 1.5
            for bet in game.hh[0]:
                if bet == -1:
                    break
                else:
                    pot += bet
            label_pot.configure(text = pot)
            street = game.street
        elif game.street == 2:
            print("street: ", street, " game.street: ", game.street)
            update_turn_img()
            for bet in game.hh[1]:
                if bet == -1:
                    break
                else:
                    pot += bet
            label_pot.configure(text = pot)
            street = game.street
        else:
            update_river_img()
            for bet in game.hh[2]:
                if bet == -1:
                    break
                else:
                    pot += bet
            label_pot.configure(text = pot)
            street = game.street
        root.update()
        if bet_villain == 0:
            labelvillainbet.configure(text = "")
        else:
            labelvillainbet.configure(text = bet_villain)
        if bet_villain == 0 and bet_hero==0:
            decideButton.configure(state=ACTIVE,text= "Bet", fg='black')
        else:
            decideButton.configure(state=ACTIVE,text= "Raise", fg='black')
        time.sleep(1)
        root.update()
    
    if bet_hero != bet_villain:
        fold_button.configure(state=ACTIVE, fg='black')
        cc_button.configure(state=ACTIVE, text= "Call",fg='black')
    else:
        cc_button.configure(state=ACTIVE, text= "Check",fg='black')
    if bet_hero == 0 and bet_villain == 0:
        decideButton.configure(state=ACTIVE, text ="Bet", fg='black')     
    else:
        decideButton.configure(state=ACTIVE, text ="Raise", fg='black')
    e.configure(state=NORMAL, fg='black')
    set_minbet()
    root.update
    return done
    

fold_button = Button(frame_action, width = 5, height = 2, text = "Fold", pady=1, command = button_fold)
cc_button = Button(frame_action, width = 5, height = 2, text = "Call", pady=1, command = button_cc)
decideButton = Button(frame_action, width = 5, height = 2, text="Raise", pady=1,command = button_betraise)
                   
fold_button.grid(row=0, column = 0)
cc_button.grid(row=0, column=1)
decideButton.grid(column = 2, row =0)


    
 

#start pokerround
done, hero_chips, villain_chips, pot, bet_hero, bet_villain= new_hand()

      
        
        


root.mainloop()