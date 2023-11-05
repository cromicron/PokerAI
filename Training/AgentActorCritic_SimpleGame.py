import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from ActorCriticNetworkSplit import ActorNetwork, CriticNetwork

class ReplayBuffer:
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
        state = state[:]
        state_ = state_[:]
        index = self.mem_cntr % self.mem_size
        
        #add information about whether the player hit the board or not.
        state.append(0) if state[4]!= state[6] else state.append(1)
        state_.append(0) if state_[4] != state_[6] else state_.append(1)
        
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


class Agent:
    def __init__(self, alpha_actor=0.00001, alpha_critic = 0.0001,n_actions=3, hidden_layers = (1024, 512)):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.actor_preflop = ActorNetwork(n_actions = 3, hidden_layers = (256,128,64,32,16,8,4))
        self.actor_postflop = ActorNetwork(n_actions = 3, hidden_layers = (256,128,64,32,16,8,4))
        self.critic = CriticNetwork()
        self.actor_preflop.compile(optimizer=Adam(learning_rate=alpha_actor))
        self.actor_postflop.compile(optimizer=Adam(learning_rate=alpha_actor))
        self.critic.compile(optimizer=Adam(learning_rate=alpha_critic), loss = 'mse')
        self.checkpoint_file_actor_preflop = 'trainedModels/ActorCriticSimple/ac_apre'
        self.checkpoint_file_actor_postflop = 'trainedModels/ActorCriticSimple/ac_apost'
        self.checkpoint_file_critic = 'trainedModels/ActorCriticSimple/ac_c'
        self.memory = ReplayBuffer(640, (18,))
        self.batch_size = 32
        
    def choose_action(self, observation):
        observation = observation[:]
        observation.append(0) if observation[4] != observation[6] else observation.append(1)
        state = tf.convert_to_tensor([observation])
        if observation[3]== 0:
            probs = self.actor_preflop(state)
        else:
            probs = self.actor_postflop(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        self.action = action

        return action.numpy()[0]
    
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
               
        
    def learn(self, state, action, reward, state_, done):
        #make board a boolean if board was hit or not
        
        if self.memory.mem_cntr >= len(self.memory.state_memory):
            

            states_tot, actions_tot, rewards_tot, states__tot, dones_tot = self.memory.sample_buffer(640)
            
            for i in range(1):
                states = states_tot[i*self.batch_size:self.batch_size+i*self.batch_size]
                actions = actions_tot[i*self.batch_size:self.batch_size+i*self.batch_size]
                rewards = rewards_tot[i*self.batch_size:self.batch_size+i*self.batch_size]
                states_ = states__tot[i*self.batch_size:self.batch_size+i*self.batch_size]
                dones = dones_tot[i*self.batch_size:self.batch_size+i*self.batch_size]
                q_eval = self.critic(states)
                #batch_index = np.arange(self.batch_size, dtype=np.int32)
                q_next = self.critic(states_)
                q_target = np.expand_dims(rewards, axis=1) + \
                                q_next*np.expand_dims(dones,axis=1)
                self.critic.train_on_batch(states, q_target)
            

                states = tf.convert_to_tensor(states)
                states_ = tf.convert_to_tensor(states_)
                rewards = tf.convert_to_tensor(rewards)

        if state[3] == 0:
            pre = True
        else:
            pre = False
        state = state[:]
        state_ = state_[:]
        state.append(0) if state[4] != state[6] else state.append(1)
        state_.append(0) if state_[4] != state_[6] else state_.append(1)
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # not fed to NN
        target = reward + self.critic(state_)*(1-int(done))
        
        
        
        with tf.GradientTape(persistent=True) as tape:
            state_value = self.critic(state)
            if pre:
                probs = self.actor_preflop(state)

            else:
                probs = self.actor_postflop(state)

            state_value_ = self.critic(state_)            
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(action)

            delta = reward - state_value + state_value_*(1-int(done)) 
            actor_loss = -log_prob*delta
            critic_loss = delta**2

        if pre:
            gradient_actor = tape.gradient(actor_loss, self.actor_preflop.trainable_variables)
            self.actor_preflop.optimizer.apply_gradients(zip(
                gradient_actor, self.actor_preflop.trainable_variables))

        else:
 
            gradient_actor = tape.gradient(actor_loss, self.actor_postflop.trainable_variables)
            self.actor_postflop.optimizer.apply_gradients(zip(
                gradient_actor, self.actor_postflop.trainable_variables))
        return actor_loss
        
        
    def save_models(self):
        print('... saving models ...')
        self.actor_preflop.save_weights(self.checkpoint_file_actor_preflop)
        self.actor_postflop.save_weights(self.checkpoint_file_actor_postflop)
        self.critic.save_weights(self.checkpoint_file_critic)

    def load_models(self):
        print('... loading models ...')
        self.actor_preflop.load_weights(self.checkpoint_file_actor_preflop)
        self.actor_postflop.load_weights(self.checkpoint_file_actor_postflop)
        self.critic.load_weights(self.checkpoint_file_critic)
    
    def print_strategy(self):
        print("--------------------------------------")
        print("preflop\n\nsb\n first in\n")

        obs=[0, 4.5, 4, 0, 3, 1.5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1,6):
            obs[4] =i
            state = tf.convert_to_tensor([obs])
            value = self.critic(state).numpy()[0][0]
            probs = self.actor_preflop(state)
            probs = list(np.round(probs.numpy()[0],3))
            print("holecard", i, "value", value, "probs", probs)
            
        print("preflop\n\nsb\n complete - push\n")

        obs=[0, 4, 0, 0, 3, 6, 0, 0.5, 4, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1,6):
            obs[4] =i
            state = tf.convert_to_tensor([obs])
            value = self.critic(state).numpy()[0][0]
            probs = self.actor_preflop(state)
            probs = list(np.round(probs.numpy()[0],3))
            print("holecard", i, "value", value, "probs", probs)

        print("\nbb\n  to complete\n")
        obs=[1, 4, 4, 0, 3, 2, 0, 0.5, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1,6):
            obs[4] =i
            state = tf.convert_to_tensor([obs])
            value = self.critic(state).numpy()[0][0]
            probs = self.actor_preflop(state)
            probs = list(np.round(probs.numpy()[0],3))
            print("holecard", i, "value", value, "probs", probs)

        print("\nbb\n  to push\n")
        obs=[1, 4, 0, 0, 3, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1,6):
            obs[4] =i
            state = tf.convert_to_tensor([obs])
            value = self.critic(state).numpy()[0][0]
            probs = self.actor_preflop(state)
            probs = list(np.round(probs.numpy()[0],3))
            print("holecard", i, "value", value, "probs", probs)


        print("--------------------------------------")
        print("postflop\n\nsb\n  to check\n")
        for board in range(1,6):#all boards
            print("board", board)
            for hole in range(1,6):
                obs=[0, 4, 4, 1, hole, 2, board, 0.5, 0, 0, -1, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                value = self.critic(state).numpy()[0][0]
                probs = self.actor_postflop(state)
                probs = list(np.round(probs.numpy()[0],3))
                print("holecard", hole, "value", value, "probs", probs)
                
        print("\n  to push\n")
        for board in range(1,6):#all boards
            print("board", board)
            for hole in range(1,6):
                obs=[0, 4, 0, 1, hole, 6, board, 0.5, 0, 4, -1, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                value = self.critic(state).numpy()[0][0]
                probs = self.actor_postflop(state)
                probs = list(np.round(probs.numpy()[0],3))
                print("holecard", hole, "value", value, "probs", probs)
        
        print("--------------------------------------")
        print("postflop\n\nbb\n  first in\n")
        for board in range(1,6):#all boards
            print("board", board)
            for hole in range(1,6):
                obs=[1, 4, 4, 1, hole, 2, board, 0.5, 0, -1, -1, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                value = self.critic(state).numpy()[0][0]
                probs = self.actor_postflop(state)
                probs = list(np.round(probs.numpy()[0],3))
                print("holecard", hole, "value", value, "probs", probs)
                
        print("\n  to push after check\n")
        for board in range(1,6):#all boards
            print("board", board)
            for hole in range(1,6):
                obs=[1, 4, 0, 1, hole, 6, board, 0.5, 0, 0, 4, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                value = self.critic(state).numpy()[0][0]
                probs = self.actor_postflop(state)
                probs = list(np.round(probs.numpy()[0],3))
                print("holecard", hole, "value", value, "probs", probs)