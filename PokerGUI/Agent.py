import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras import backend as K
from StrengthEvaluator2 import StrengthEvaluator as sEval

evaluator = sEval()
class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), 
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal



def build_dqn(lr, n_actions, input_dims, stacksize, layerSizes):
    model = keras.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=-1., input_shape=input_dims))
    
    for size in layerSizes:
        model.add(keras.layers.Dense(size))
        model.add(keras.layers.LeakyReLU(alpha=0.05))
        
    model.add(keras.layers.Dense(n_actions,activation=None))
    model.add(keras.layers.Lambda(lambda x: K.tanh(0.1*x)*stacksize))

    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return model
class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims,
                stacksize, epsilon_dec=1e-4, epsilon_end=0.02,
                mem_size=1000000, fname='PokerNewAi.h5', handEval = True, layerSizes = (16,8)):
        self.handEval = handEval
        if self.handEval: #old version of this class has not had a hand evaluator yet
            input_dims = (input_dims[0]+ 2,)

        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, stacksize, layerSizes)
        self.episodes_learned = 0
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        #if no action has been performed on the flop/ turn/river yet, evaluate handstrength
        if self.handEval:
            if observation[13] == -1: #no preflop_action for second player to act yet. (villain might have acted)
                if observation[6] == -1:
                    hand = [(observation[4], 0), (observation[5], 0)]
                else:
                    hand = [(observation[4], 0), (observation[5], 1)]
                evaluator.evaluate(hand)

            elif observation[33] == -1:
                hand = [(observation[4], observation[11]), (observation[5], observation[12]),(observation[-10], observation[-9]), (observation[-8], observation[-7]),(observation[-6], observation[-5])]
                evaluator.evaluate(hand)

            elif observation[53] == -1:
                hand = [(observation[4], observation[11]), (observation[5], observation[12]),(observation[-10], observation[-9]), (observation[-8], observation[-7]),(observation[-6], observation[-5]),(observation[-4], observation[-3])]
                evaluator.evaluate(hand)
            elif observation[73] == -1:
                hand = [(observation[4], observation[11]), (observation[5], observation[12]),(observation[-10], observation[-9]), (observation[-8], observation[-7]),(observation[-6], observation[-5]),(observation[-4], observation[-3]),(observation[-2], observation[-1])]

            self.pwin = evaluator.pwin
            self.ploose = evaluator.ploose
            self.pwinAvg = evaluator.pwinAvg
            self.plooseAvg = evaluator.plooseAvg
            self.stdWinAvg =evaluator.stdWinAvg
            observation = np.append(observation,[self.pwin, self.ploose,self.pwinAvg,self.plooseAvg,self.stdWinAvg])

        state = np.array([observation])
        actions = self.q_eval.predict(state)

        if np.random.random() < self.epsilon:
            #half the time choose action by weight
            if np.random.random() < 0.5:
                weights=actions-actions.min()
                action = random.choices(
                population=range(12),
                weights=list(weights[0]),
                k=1)[0]
            #half the time chose a random action
            else:
                if observation[8] != observation[9]: # check if bets are eqal, so fold is not tried.
                    action = np.random.randint(0,3) #make fold, call/check and bet/raise equally likely
                    if action == 2:
                        action = np.random.randint(2,12)
                else: #bets are balanced, so fold should not be chosen.
                    action = np.random.randint(1,3)
                    if action ==2:
                        action = np.random.randint(2,12)

        else:
            action = np.argmax(actions)


        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)


        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones


        self.q_eval.train_on_batch(states, q_target)
        self.q_eval.evaluate(q_eval, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min
        
    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file,custom_objects={"K": K}) 
    
