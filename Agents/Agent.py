import copy
from itertools import combinations
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

deck = []
for suit in range(4):
    for value in range(2,15):
        deck.append([value, suit])

cards =[[card[0], card[1]] for card in deck]
cards.sort()
hands = list(combinations(cards,2))

hands_array = np.array(hands).reshape(1326,4)


class Range:
    def __init__(self,prob_array):
        if type(prob_array)!= np.ndarray:
            raise TypeError("prob_array must be a np.array")
        if len(prob_array) != 1326:
            raise ValueError("prob_array must have length of 1326")
        self.range = prob_array



class Agent:
    def __init__(self,path_action = "action.h5", path_bet="bet.h5"):
        self.range_hero = None
        self.range_villain = None
        self.path_action_network = path_action
        self.path_bet_network = path_bet

    def create_action_network(self, input_dims, layers= (128,64,32,16,8,4), lr = 0.0001):
        self.action_model = keras.Sequential()
        self.action_model.add(keras.layers.Dense(layers[0], input_dim=input_dims, activation = 'LeakyReLU'))
        for i in range(1,len(layers)):
            self.action_model.add(keras.layers.Dense(layers[i], activation = 'LeakyReLU'))
        self.action_model.add(keras.layers.Dense(3,activation='softmax'))
        self.action_model.compile(optimizer=Adam(learning_rate=lr), loss='kl_divergence',run_eagerly=True)

    def create_bet_network(self, input_dims, layers= (128,64,32,16,8,4), lr = 0.0001):
        self.bet_model = keras.Sequential()
        self.bet_model.add(keras.layers.Dense(layers[0], input_dim=input_dims, activation = 'LeakyReLU'))
        for i in range(1,len(layers)):
            self.bet_model.add(keras.layers.Dense(layers[i], activation = 'LeakyReLU'))
        self.bet_model.add(keras.layers.Dense(1,activation='sigmoid'))
        self.bet_model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')

    def get_action_probs(self, observation) -> np.array:
        probs = self.action_model(np.array([observation]))
        return np.array(probs)[0]

    def get_bet_probs(self, observation):
        return np.array(self.bet_model(np.array([observation])))[0]

    def save_networks(self, networks = (0,1)):
        if 0 in networks:
            self.action_model.save(self.path_action_network)
        if 1 in networks:
            self.bet_model.save(self.path_bet_network)

    def initialize_ranges(self,holecards):
        holecards.sort()
        val_1 = holecards[0][0]
        suit_1 = holecards[0][1]
        val_2 = holecards[1][0]
        suit_2 = holecards[1][1]
        prob_array = np.full((1326),1/1326)
        prob_array_villain = np.full((1326),1/1324)
        self.range_hero = Range(prob_array) #from the perspective of the other player - all hands equally likely
        bool_array = np.all((hands_array[:,0:2]==[val_1,suit_1])|(hands_array[:,2:4]==[val_2,suit_2]),axis=1)
        prob_array_vil=(1-bool_array).astype(int)*prob_array_villain
        self.range_villain = Range(prob_array_vil)

    def policy_range(self,observation):
        # returns actions probabilites for all cards
        obs_array = np.full((1326,len(observation)),observation)
        obs_array[:,5:9]=hands_array
        return self.action_model(obs_array)

    def update_range(self,observation,action,perspective = 'hero'):
        policy = self.policy_range(observation)
        if perspective=='hero':
            numerator = tf.math.multiply(policy[:,action],self.range_villain.range)
            denominator = tf.math.reduce_sum(numerator)
            self.range_villain.range = np.array(numerator/denominator)

        else:
            if perspective != 'villain':
                raise ValueError("perspective must be either hero or villain")
            numerator = tf.math.multiply(policy[:,action],self.range_hero.range)
            denominator = tf.math.reduce_sum(numerator)
            self.range_hero.range = np.array(numerator/denominator)

    def simulate_hand(self, game_state):
        game = copy.deepcopy(game_state)
        postition_hero = game_state.next[-1].position
        while not game_state.done:
            if game.netx[-1].position == postition_hero:
                observation = game.get_observation()
                legal_actions = game.get_legal_actions()
                action_probs = self.get_action_probs(observation)

                while action not in legal_actions:
                    action = self.choose_action(observation)
                self.update_range(observation,perspective='villain')
                if action == 2:
                    bet = self.choose_bet(observation)




