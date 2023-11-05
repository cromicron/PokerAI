# this agent is capable of updating villains ranges according to his own policy and uses these ranges to simulate outcomes of his implemented action
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from ActorCriticNetworkSplit import ActorNetwork, CriticNetwork
import copy


class ReplayBuffer:
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward):
        state = state[:]
        index = self.mem_cntr % self.mem_size

        # add information about whether the player hit the board or not.
        state.append(0) if state[4] != state[6] else state.append(1)

        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]

        return states, actions, rewards


class Agent:
    def __init__(self, alpha_actor=0.00001, alpha_critic=0.0001, n_actions=3, hidden_layers=(1024, 512)):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.actor_preflop = ActorNetwork(n_actions=3, hidden_layers=(256, 128, 64, 32, 16, 8, 4))
        self.actor_postflop = ActorNetwork(n_actions=3, hidden_layers=(256, 128, 64, 32, 16, 8, 4))
        self.critic = CriticNetwork()
        self.actor_preflop.compile(optimizer=Adam(learning_rate=alpha_actor))
        self.actor_postflop.compile(optimizer=Adam(learning_rate=alpha_actor))
        self.critic.compile(optimizer=Adam(learning_rate=alpha_critic), loss='mse')
        self.checkpoint_file_actor_preflop = 'trainedModels/ActorCriticSimpleRange/ac_apre_range'
        self.checkpoint_file_actor_postflop = 'trainedModels/ActorCriticSimpleRange/ac_apost_range'
        self.checkpoint_file_critic = 'trainedModels/ActorCriticSimpleRange/ac_c_range'
        self.memory = ReplayBuffer(640, (18,))
        self.batch_size = 32
        self.saw_flop = [False, False]
        self.ranges_vil = [[], []]

    def choose_action(self, observation):
        observation = observation[:]
        observation.append(0) if observation[4] != observation[6] else observation.append(1)
        state = tf.convert_to_tensor([observation])
        if observation[3] == 0:
            probs = self.actor_preflop(state)
        else:
            probs = self.actor_postflop(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        self.action = action

        return action.numpy()[0]

    def store_transition(self, state, action, reward):
        self.memory.store_transition(state, action, reward)

    def set_range_vil(self, player, observation):
        for i in range(1, 6):
            if i == observation[4]:
                self.ranges_vil[player].append(1 / 9)
            else:
                self.ranges_vil[player].append(2 / 9)

    def update_range_action(self, player, observation_villain, action_vil):
        # should be used after villain has chosen an action
        states = []
        for card in range(1, 6):
            observation = observation_villain[:]
            observation[4] = card
            observation.append(0) if observation[4] != observation[6] else observation.append(1)
            states.append(observation)

        states = tf.convert_to_tensor(states)
        p_cards_prior = np.array(self.ranges_vil[player])
        p_action_given_card = self.actor_preflop(states)[:, action_vil] if observation[3] == 0 else self.actor_postflop(
            states)[:, action_vil]
        p_action = tf.reduce_sum(p_action_given_card * p_cards_prior)
        posterior = np.array(p_cards_prior * p_action_given_card / p_action)
        self.ranges_vil[player] = list(posterior / posterior.sum())
        return

    def update_range_flop(self, player, hole_hero, board):
        # should be used after a new board card has been dealt
        if hole_hero == board:
            self.ranges_vil[player][board - 1] = 0
        else:
            self.ranges_vil[player][board - 1] *= 0.5
        # normalize to 1
        posterior = np.array(self.ranges_vil[player]) / (np.array(self.ranges_vil[player]).sum())
        self.saw_flop[player - 1] = True
        self.ranges_vil[player] = list(posterior)
        return

    def simulate(self, current_game,
                 n_simulations):  # take in current game object, copies it and simulates runouts. returns avg rewards.

        total_reward_0, total_reward_1 = 0, 0

        # after each simulation, we have to reset ranges and saw flop
        ranges_vil = copy.copy(self.ranges_vil)
        saw_flop = copy.copy(self.saw_flop)

        for k in range(n_simulations):
            game = copy.deepcopy(current_game)
            self.ranges_vil = ranges_vil
            self.saw_flop = saw_flop
            while not game.done:
                next_to_act = game.next_to_act[0]
                # check if there is already an action for the player, so we do a learning step before taking a new action.
                game.create_observation(next_to_act)
                action = self.choose_action(game.observations[next_to_act])

                if game.observations[next_to_act][6] != 0 and not self.saw_flop[next_to_act]:
                    hole_hero = game.hole_0 if next_to_act == 1 else game.hole_1
                    self.update_range_flop(1 - next_to_act, hole_hero, game.observations[next_to_act][6])
                self.update_range_action(1 - next_to_act, game.observations[next_to_act], action)

                game.implement_action(next_to_act, action)

                if not game.done:  # game can be done, if action was fold or the game ended because it was the last action.

                    next_to_act = game.next_to_act[0]
                    game.create_observation(next_to_act)
                    action = self.choose_action(game.observations[next_to_act])

                    if game.observations[next_to_act][6] != 0 and not self.saw_flop[next_to_act]:
                        hole_hero = game.hole_0 if next_to_act == 1 else game.hole_1
                        self.update_range_flop(1 - next_to_act, hole_hero, game.observations[next_to_act][6])

                    self.update_range_action(1 - next_to_act, game.observations[next_to_act], action)
                    game.implement_action(next_to_act, action)

            total_reward_0 += game.stacks[0] - 4.5 if game.position_0 == 0 else game.stacks[0] - 4
            total_reward_1 += game.stacks[1] - 4.5 if game.position_1 == 0 else game.stacks[1] - 4

        avg_reward_0 = total_reward_0 / n_simulations
        avg_reward_1 = total_reward_1 / n_simulations
        #reset ranges to initial ones
        self.ranges_vil = ranges_vil
        self.saw_flop = saw_flop
        return avg_reward_0, avg_reward_1

    def learn(self, state, action, reward):
        # make board a boolean if board was hit or not

        if self.memory.mem_cntr >= len(self.memory.state_memory):

            states_tot, actions_tot, rewards_tot = self.memory.sample_buffer(640)

            for i in range(1):
                states = states_tot[i * self.batch_size:self.batch_size + i * self.batch_size]
                actions = actions_tot[i * self.batch_size:self.batch_size + i * self.batch_size]
                rewards = rewards_tot[i * self.batch_size:self.batch_size + i * self.batch_size]

                q_eval = self.critic(states)
                # batch_index = np.arange(self.batch_size, dtype=np.int32)

                self.critic.train_on_batch(states, rewards)

                states = tf.convert_to_tensor(states)
                rewards = tf.convert_to_tensor(rewards)

        if state[3] == 0:
            pre = True
        else:
            pre = False
        state = state[:]

        state.append(0) if state[4] != state[6] else state.append(1)
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)  # not fed to NN

        with tf.GradientTape(persistent=True) as tape:
            state_value = self.critic(state)
            if pre:
                probs = self.actor_preflop(state)
            else:
                probs = self.actor_postflop(state)
            state_value = tf.squeeze(state_value)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(action)

            delta = reward - state_value
            actor_loss = -log_prob * delta
            # critic_loss = delta**2

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
        self.actor_preflop.save_weights(self.checkpoint_file_actor_postflop)
        self.critic.save_weights(self.checkpoint_file_critic)

    def load_models(self):
        print('... loading models ...')
        self.actor_preflop.load_weights(self.checkpoint_file_actor_preflop)
        self.actor_postflop.load_weights(self.checkpoint_file_actor_postflop)
        self.critic.load_weights(self.checkpoint_file_critic)

    def print_strategy(self):
        print("--------------------------------------")
        print("preflop\n\nsb\n first in\n")

        obs = [0, 4.5, 4, 0, 3, 1.5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1, 6):
            obs[4] = i
            state = tf.convert_to_tensor([obs])
            value = self.critic(state).numpy()[0][0]
            probs = self.actor_preflop(state)
            probs = list(np.round(probs.numpy()[0], 3))
            print("holecard", i, "value", value, "probs", probs)

        print("preflop\n\nsb\n complete - push\n")

        obs = [0, 4, 0, 0, 3, 6, 0, 0.5, 4, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1, 6):
            obs[4] = i
            state = tf.convert_to_tensor([obs])
            value = self.critic(state).numpy()[0][0]
            probs = self.actor_preflop(state)
            probs = list(np.round(probs.numpy()[0], 3))
            print("holecard", i, "value", value, "probs", probs)

        print("\nbb\n  to complete\n")
        obs = [1, 4, 4, 0, 3, 2, 0, 0.5, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1, 6):
            obs[4] = i
            state = tf.convert_to_tensor([obs])
            value = self.critic(state).numpy()[0][0]
            probs = self.actor_preflop(state)
            probs = list(np.round(probs.numpy()[0], 3))
            print("holecard", i, "value", value, "probs", probs)

        print("\nbb\n  to push\n")
        obs = [1, 4, 0, 0, 3, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1, 6):
            obs[4] = i
            state = tf.convert_to_tensor([obs])
            value = self.critic(state).numpy()[0][0]
            probs = self.actor_preflop(state)
            probs = list(np.round(probs.numpy()[0], 3))
            print("holecard", i, "value", value, "probs", probs)

        print("--------------------------------------")
        print("postflop\n\nsb\n  to check\n")
        for board in range(1, 6):  # all boards
            print("board", board)
            for hole in range(1, 6):
                obs = [0, 4, 4, 1, hole, 2, board, 0.5, 0, 0, -1, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                value = self.critic(state).numpy()[0][0]
                probs = self.actor_postflop(state)
                probs = list(np.round(probs.numpy()[0], 3))
                print("holecard", hole, "value", value, "probs", probs)

        print("\n  to push\n")
        for board in range(1, 6):  # all boards
            print("board", board)
            for hole in range(1, 6):
                obs = [0, 4, 0, 1, hole, 6, board, 0.5, 0, 4, -1, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                value = self.critic(state).numpy()[0][0]
                probs = self.actor_postflop(state)
                probs = list(np.round(probs.numpy()[0], 3))
                print("holecard", hole, "value", value, "probs", probs)

        print("--------------------------------------")
        print("postflop\n\nbb\n  first in\n")
        for board in range(1, 6):  # all boards
            print("board", board)
            for hole in range(1, 6):
                obs = [1, 4, 4, 1, hole, 2, board, 0.5, 0, -1, -1, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                value = self.critic(state).numpy()[0][0]
                probs = self.actor_postflop(state)
                probs = list(np.round(probs.numpy()[0], 3))
                print("holecard", hole, "value", value, "probs", probs)

        print("\n  to push after check\n")
        for board in range(1, 6):  # all boards
            print("board", board)
            for hole in range(1, 6):
                obs = [1, 4, 0, 1, hole, 6, board, 0.5, 0, 0, 4, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                value = self.critic(state).numpy()[0][0]
                probs = self.actor_postflop(state)
                probs = list(np.round(probs.numpy()[0], 3))
                print("holecard", hole, "value", value, "probs", probs)
