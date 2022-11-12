# this agent is capable of updating villains ranges according to his own policy and uses these ranges to simulate outcomes of his implemented action
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras import backend as K
import tensorflow_probability as tfp
import copy


class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)

        self.policy_memory = np.zeros((self.mem_size, *(n_actions,)), dtype=np.float32)

    def store_sample(self, state, policy):
        index = self.mem_cntr % self.mem_size
        # add information about whether the player hit the board or not.
        state.append(0) if state[4] != state[6] else state.append(1)

        self.state_memory[index] = state
        self.policy_memory[index] = policy
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        policies = self.policy_memory[batch]

        return states, policies


def create_network(self, n_actions, layer_sizes, lr):
    model = keras.Sequential()

    for size in layer_sizes:
        model.add(keras.layers.Dense(size))
        model.add(keras.layers.LeakyReLU(alpha=0.05))

    model.add(keras.layers.Dense(n_actions, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='SparseCategoricalCrossentropy')

    return model


class Agent:
    def __init__(self, lr=0.00001, n_actions=3, hidden_layers=(64, 32, 16, 8), policy_networks = []):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.actor = create_network(self, n_actions=3, layer_sizes=hidden_layers, lr=lr)
        self.checkpoint_file = 'trainedModels/Simulation/sim'
        self.memory = ReplayBuffer(4000, (18,), n_actions)
        self.batch_size = 32
        self.saw_flop = [False, False]
        self.ranges_vil = [[], []]
        self.policy_networks = policy_networks

    def choose_action(self, observation):
        observation = observation[:]
        observation.append(0) if observation[4] != observation[6] else observation.append(1)
        state = tf.convert_to_tensor([observation])
        probs = self.actor(state)
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        return action.numpy()[0]

    def store_sample(self, state, policy):
        self.memory.store_sample(state, policy)

    def set_range_vil(self, player, holecard, hole_known= True):
        self.ranges_vil[player] = []
        if not hole_known or holecard < 1: #not known or invalid holecard
            for i in range(1, 6):
                self.ranges_vil[player].append(1 / 5)

        else:
            for i in range(1, 6):
                if i == holecard:
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
        if self.policy_networks == []:
            p_action_given_card = self.actor(states)[:, action_vil]
        else:

        p_action = tf.reduce_sum(p_action_given_card * p_cards_prior)
        posterior = np.array(p_cards_prior * p_action_given_card / p_action)
        self.ranges_vil[player] = list(posterior / posterior.sum())

    def update_range_flop(self, player, hole_hero, board, hole_known = True):
        # should be used after a new board card has been dealt. The hole known
        # argument tracks if the holecard of the player to simulate expected
        # range about his opponent is known. If we want to simulate an opponent
        #calculating hero's range, we cannot let him know his own holecard.
        if hole_known:
            if hole_hero == board:
                self.ranges_vil[player][board - 1] = 0
            else:
                self.ranges_vil[player][board - 1] *= 0.5
        else:
            self.ranges_vil[player][board - 1] *= 0.5
        # normalize to 1
        posterior = np.array(self.ranges_vil[player]) / (np.array(self.ranges_vil[player]).sum())
        self.saw_flop[player - 1] = True
        self.ranges_vil[player] = list(posterior)

    def simulate(self, current_game,
                 n_simulations,
                 hero):  # take in current game object, copies it and simulates runouts. returns avg rewards.
        # both players have ranges. We take the range that is ascribed to the other player to simulate the runouts,
        # NOT the true current game. The simulate function is called after hero's action is implemented, to get an
        # estimate of the expected reward of the action.
        # we must therefore draw new cards for villain and if the flop is not yet known randomly draw a flop

        total_reward_0, total_reward_1 = 0, 0

        # after each simulation, we have to reset ranges and saw flop
        ranges_vil = copy.copy(self.ranges_vil)
        saw_flop = copy.copy(self.saw_flop)

        for k in range(n_simulations):

            if hero == 0:
                hand_villain = current_game.hole_1
            else:
                hand_villain = current_game.hole_0

            game = copy.deepcopy(current_game)
            # draw new hand_villain from range
            new_hand_vil = random.choices([1, 2, 3, 4, 5], self.ranges_vil[hero])[0]
            # exchange cards for villain
            if new_hand_vil != hand_villain:
                index_new_hand_vil = game.deck[2:].index(new_hand_vil) + 2
                if hero == 0:
                    game.hole_1 = new_hand_vil
                    game.deck[index_new_hand_vil], game.deck[1] = hand_villain, new_hand_vil
                else:
                    game.hole_0 = new_hand_vil
                    game.deck[index_new_hand_vil], game.deck[0] = hand_villain, new_hand_vil

            self.ranges_vil = ranges_vil
            self.saw_flop = saw_flop

            while not game.done:
                next_to_act = game.next_to_act[0]
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
            game.deck = current_game.deck
        avg_reward_0 = total_reward_0 / n_simulations
        avg_reward_1 = total_reward_1 / n_simulations
        # reset ranges to initial ones
        self.ranges_vil = ranges_vil
        self.saw_flop = saw_flop
        return avg_reward_0, avg_reward_1

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, policy = self.memory.sample_buffer(self.batch_size)
        self.actor.train_on_batch(states, policy)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.checkpoint_file)

    def print_strategy(self):
        print("--------------------------------------")
        print("preflop\n\nsb\n first in\n")

        obs = [0, 4.5, 4, 0, 3, 1.5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1, 6):
            obs[4] = i
            state = tf.convert_to_tensor([obs])
            probs = self.actor(state)
            probs = list(np.round(probs.numpy()[0], 3))
            print("holecard", i, "probs", probs)

        print("preflop\n\nsb\n complete - push\n")

        obs = [0, 4, 0, 0, 3, 6, 0, 0.5, 4, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1, 6):
            obs[4] = i
            state = tf.convert_to_tensor([obs])
            probs = self.actor(state)
            probs = list(np.round(probs.numpy()[0], 3))
            print("holecard", i, "probs", probs)

        print("\nbb\n  to complete\n")
        obs = [1, 4, 4, 0, 3, 2, 0, 0.5, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1, 6):
            obs[4] = i
            state = tf.convert_to_tensor([obs])
            probs = self.actor(state)
            probs = list(np.round(probs.numpy()[0], 3))
            print("holecard", i, "probs", probs)

        print("\nbb\n  to push\n")
        obs = [1, 4, 0, 0, 3, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        obs.append(0) if obs[4] != obs[6] else obs.append(1)
        for i in range(1, 6):
            obs[4] = i
            state = tf.convert_to_tensor([obs])
            probs = self.actor(state)
            probs = list(np.round(probs.numpy()[0], 3))
            print("holecard", i, "probs", probs)

        print("--------------------------------------")
        print("postflop\n\nsb\n  to check\n")
        for board in range(1, 6):  # all boards
            print("board", board)
            for hole in range(1, 6):
                obs = [0, 4, 4, 1, hole, 2, board, 0.5, 0, 0, -1, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                probs = self.actor(state)
                probs = list(np.round(probs.numpy()[0], 3))
                print("holecard", hole, "probs", probs)

        print("\n  to push\n")
        for board in range(1, 6):  # all boards
            print("board", board)
            for hole in range(1, 6):
                obs = [0, 4, 0, 1, hole, 6, board, 0.5, 0, 4, -1, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                probs = self.actor(state)
                probs = list(np.round(probs.numpy()[0], 3))
                print("holecard", hole, "probs", probs)

        print("--------------------------------------")
        print("postflop\n\nbb\n  first in\n")
        for board in range(1, 6):  # all boards
            print("board", board)
            for hole in range(1, 6):
                obs = [1, 4, 4, 1, hole, 2, board, 0.5, 0, -1, -1, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                probs = self.actor(state)
                probs = list(np.round(probs.numpy()[0], 3))
                print("holecard", hole, "probs", probs)

        print("\n  to push after check\n")
        for board in range(1, 6):  # all boards
            print("board", board)
            for hole in range(1, 6):
                obs = [1, 4, 0, 1, hole, 6, board, 0.5, 0, 0, 4, -1, -1, -1, -1, -1, -1]
                obs.append(0) if obs[4] != obs[6] else obs.append(1)
                state = tf.convert_to_tensor([obs])
                probs = self.actor(state)
                probs = list(np.round(probs.numpy()[0], 3))
                print("holecard", hole, "probs", probs)
