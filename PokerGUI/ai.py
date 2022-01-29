#Model and Agent from: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_tf2.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras import backend as K

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



def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=-1., input_shape=input_dims))
    model.add(keras.layers.Dense(fc1_dims))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dense(fc2_dims))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU (alpha=0.05))
    model.add(keras.layers.Dense(n_actions,activation=None))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Lambda(lambda x: K.tanh(0.1*x)*20)) #the input to this layer shoudl be between -3 and 3, so that the max/min outputs are .995

    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return model
class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=1e-4, epsilon_end=0.02,
                mem_size=1000000, fname='HUPoker.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 51,26)
        self.total_loss = 0 #keep track of the total loss
        self.episodes_learned = 0
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        state = np.array([observation])
        actions = self.q_eval.predict(state)

        if np.random.random() < self.epsilon:
            weights=actions-actions.min()
            action = random.choices(
            population=range(12),
            weights=list(weights[0]),
            k=1)[0]

        else:
            action = np.argmax(actions)


        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(len(self.memory.state_memory))

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)
        batch_index = np.arange(len(self.memory.state_memory), dtype=np.int32)

        q_target = np.copy(q_eval)


        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones
        
        # we know the exact q-values for folds. You lose exactly what you have put in the pot thus far.

        for i in batch_index:
            q_target[i, 0] = states[i, 1] -19.5 if states[i, 0] == 0 else states[i,2] -19
            #sometimes we know that qs for different actions are the same. When villain is all- all qs except for fold are equal.they are all the same as allin, so we update q for all actions.
            # this is the last action everytime, so all get the reward.

            if ((states[i, 0] == 0 and states[i, 2] ==0) or (states[i, 0] == 1 and states[i, 1]==0)):
                q_target[i, 2:] = -20
                if rewards[i] != 0:
                    q_target[i, 1] = rewards[i]
            if states[i,8] == states[i,9]:
                q_target[i, 0] = -20
 

        
        self.q_eval.fit(states, q_target, batch_size=400, epochs=4)


    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file)
        
agent= Agent(gamma=1, epsilon=0, lr=0.05,
        input_dims=(102,),
        n_actions=12, mem_size=1, batch_size=64,
        epsilon_end=0.05,
        fname='PokerMasked.h5')
agent.load_model()