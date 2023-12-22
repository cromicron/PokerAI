import copy
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow_probability import distributions
from tensorflow.keras.optimizers import Adam
import numpy as np
from PokerGame.NLHoldem import Card, value_dict, suit_dict
from Agents.networks import q_network_actor, actor_with_regret, bet_size_actor, q_continuous, generate_maxima_finder
import logging

# Set the threshold for what messages will be logged
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# Define the total number of card values and suits
NUM_CARD_VALUES = 13  # From 2 to Ace
NUM_SUITS = 4  # Clubs, Diamonds, Hearts, Spades
STREET_DICT = {"PreFlop": 0, "Flop": 1, "Turn": 2, "River": 3}
value_dict_flipped = {value: key for key, value in value_dict.items()}
value_dict_flipped.update({str(i): i for i in range(2, 10)})
suit_dict_flipped = {value: key for key, value in suit_dict.items()}
class EpisodeBuffer:
    def __init__(self, input_size: int, output_size: int, buffer_size: int = 1000):
        self.observations = np.empty((0, input_size))
        self.actions = np.empty((0, 1))
        self.evs = np.empty((0, output_size))
        self.observations_bet = np.empty((0, input_size))
        self.bet_size_fractions = np.empty((0, 1))
        self.polynomials = np.empty((0, 5))
        self.q_betsize = np.empty((0, 1))
        self.buffer_size = buffer_size
class Agent:
    def __init__(self, batch_size = 32, input_shape=(225,)):
        self.actor = actor_with_regret(input_dims=input_shape)
        self.q_action = q_network_actor(input_dims=input_shape)
        self.betsize_network = bet_size_actor(input_shape=input_shape)
        self.q_betsize = q_continuous(input_dims=input_shape)
        self.optimizer = Adam(learning_rate=0.0001)
        self.episode = None
        self.reset_episode()
        self.episode_buffer = EpisodeBuffer(input_size=input_shape[0], output_size=3)
        self.episodes_q_trained = 0
        self.episodes_q_cont_trained = 0
        self.batch_size = batch_size
        self.maxima_finder = generate_maxima_finder(4, (0,1))

    def reset_episode(self):
        self.episode = {"inputs": [], "actions": [], "bet_size_fractions": []}
    def add_to_episode(self, inputs, action, bet_size_frac):
        self.episode["inputs"].append(inputs)
        self.episode["actions"].append(action)
        if bet_size_frac is None:
            bet_size_frac = 0
        self.episode["bet_size_fractions"].append(bet_size_frac)

    def consolidate_episode(self, reward):
        """
        creates training samples for q-network
        """
        states = np.array(self.episode["inputs"])
        actions = self.episode["actions"]
        qs = self.q_action(states)
        action_probs = self.actor(states)
        evs_policy = tf.reduce_mean(qs*action_probs, axis=1)
        rewards = tf.concat([evs_policy[1:], [reward]], axis=0)
        mask = tf.cast(tf.one_hot(actions, depth = 3, dtype=tf.int32), tf.bool)
        qs = tf.where(mask, rewards[:, tf.newaxis], qs)

        self.episode_buffer.observations = np.vstack((self.episode_buffer.observations, states))
        self.episode_buffer.evs = np.vstack((self.episode_buffer.evs, qs))
        self.episode_buffer.actions = np.vstack((self.episode_buffer.actions, np.array(actions).reshape(-1,1)))
        buffer_overflow = self.episode_buffer.observations.shape[0] - self.episode_buffer.buffer_size
        if buffer_overflow > 0:
            self.episode_buffer.observations = self.episode_buffer.observations[buffer_overflow:]
            self.episode_buffer.evs = self.episode_buffer.evs[buffer_overflow:]
            self.episode_buffer.actions = self.episode_buffer.actions[buffer_overflow:]
        # information for training betsize.
        if 2 in actions:
            bet_size_fractions = np.array(self.episode["bet_size_fractions"])
            mask_bet = bet_size_fractions!=0
            bet_size_fractions = bet_size_fractions[mask_bet]
            observations_betsize = states[mask_bet]
            qs_bet = qs.numpy()[mask_bet, 2].reshape(-1, 1) # get value of action taken
            self.episode_buffer.bet_size_fractions = np.vstack(
                (self.episode_buffer.bet_size_fractions, bet_size_fractions.reshape(-1,1))
            )
            self.episode_buffer.observations_bet = np.vstack(
                (self.episode_buffer.observations_bet, observations_betsize)
            )
            self.episode_buffer.q_betsize = np.vstack(
                (self.episode_buffer.q_betsize, qs_bet)
            )
            buffer_overflow_betsize = self.episode_buffer.q_betsize.shape[0] - self.episode_buffer.buffer_size
            if buffer_overflow_betsize > 0:
                self.episode_buffer.q_betsize = self.episode_buffer.q_betsize[buffer_overflow_betsize:]
                self.episode_buffer.bet_size_fractions = self.episode_buffer.bet_size_fractions[buffer_overflow:]
                self.episode_buffer.observations_bet = self.episode_buffer.observations_bet[buffer_overflow:]
        self.reset_episode()

    def add_observation_to_q_buffer(self, observation, action, reward):
        qs = self.q_action(observation).numpy()
        qs[0, action] = reward
        self.episode_buffer.observations = np.vstack((self.episode_buffer.observations, observation))
        self.episode_buffer.evs = np.vstack((self.episode_buffer.evs, qs))
        self.episode_buffer.actions = np.vstack((self.episode_buffer.actions, action))

    def train_on_episode(self):
        buffer_size = self.episode_buffer.evs.shape[0]
        if buffer_size > self.batch_size:
            sample_indices = np.random.choice(buffer_size, self.batch_size, replace=False)
            observations = self.episode_buffer.observations[sample_indices]
            evs = self.episode_buffer.evs[sample_indices]
            self.q_action.train_on_batch(observations, evs)
            self.episodes_q_trained += 1
            if self.episodes_q_trained >= 500:
                evs = self.q_action(observations)
                self.actor.train_on_batch(observations, evs)
            buffer_size_bet = self.episode_buffer.bet_size_fractions.shape[0]
            if buffer_size_bet > 32:
                sample_indices_bet = np.random.choice(buffer_size_bet, self.batch_size, replace=False)
                observations = self.episode_buffer.observations_bet[sample_indices_bet]
                bet_fracs = self.episode_buffer.bet_size_fractions[sample_indices_bet]
                evs_bet = self.episode_buffer.q_betsize[sample_indices_bet]
                labels = np.concatenate([evs_bet, bet_fracs], axis=1)
                self.q_betsize.train_on_batch(observations, labels)
                self.episodes_q_cont_trained += 1
                if self.episodes_q_cont_trained > 50:
                    polynomials = self.q_betsize.predict(observations, verbose=0)
                    max_ev = self.maxima_finder(polynomials)[1].reshape(-1,1)
                    labels = np.concatenate([max_ev, polynomials],axis=1)
                    self.betsize_network.train_on_batch(observations, labels)

    def save_model(self, name_actor = "actor_model", name_q = "q_model"):
        self.actor.save("saved_models/"+ name_actor)
        self.q_action.save("saved_models/" + name_q)
    def one_hot_encode_hand(self, card1, card2):
        # Calculate the unique index for each card based on value and suit

        card1_index = (card1.value - 2) * 4 + card1.suit  # Since value is 2-14, subtract 2 to start index at 0
        card2_index = (card2.value - 2) * 4 + card2.suit

        # Create a two-hot encoded vector of size 52 for the deck of cards
        encoded_hand = np.zeros(52, dtype=int)
        encoded_hand[card1_index] = 1
        encoded_hand[card2_index] = 1

        # we should also encode if the hand is suited or not so the model doesn't need to learn it
        suited = 1 if card1.suit == card2.suit else 0

        # To help the model learn more quickly, we are going to tell it the current hand strength
        # Strengths are floating points for high cards 0.1413 (AKh)
        strength = 1 if card1.value == card2.value else 0
        add_strength = 0.01*card1.value if card1.value == card2.value else 0.01* max(card1.value, card2.value)+ 0.0001*min(card1.value, card2.value)
        encoded_strength = np.array([strength + add_strength])/4
        encoded_starting_hand = np.concatenate([encoded_hand, np.array([suited]), encoded_strength])
        return encoded_starting_hand
    def encode_street(self, cards):
        encoded_street = np.zeros(52, dtype=int)
        for card in cards:
            value = value_dict_flipped[card[0]]
            suit = suit_dict_flipped[card[1]]
            index = (value - 2) * 4 + suit
            encoded_street[index] = 1
        return encoded_street
    def create_state(self, cards, position, hand_history, previous_state=None):

        if position == 0:

            player_id_self = hand_history["Blinds"]["Small"]["Player"]
            player_id_other = hand_history["Blinds"]["Big"]["Player"]
        else:
            player_id_self = hand_history["Blinds"]["Big"]["Player"]
            player_id_other = hand_history["Blinds"]["Small"]["Player"]
        if previous_state is None:
            street = 0
            starting_stacks = hand_history["Players"]
            if starting_stacks[0]["Player"] == player_id_self:
                starting_stack_self = starting_stacks[0]["Stack"]
                starting_stack_other = starting_stacks[1]["Stack"]
            else:
                starting_stack_self = starting_stacks[1]["Stack"]
                starting_stack_other = starting_stacks[0]["Stack"]
            if position == 0:
                bet_size_self = 1
                bet_size_other = 2
                call_check_other_0 = 0
                bet_raise_other_0 = 0


            else:
                bet_size_self = 2
                action_other = hand_history["Actions"]["PreFlop"][0]
                bet_size_other = 2 if action_other["Action"] == "Call" else action_other["Amount"]
                call_check_other_0 = 1 if action_other["Action"] == "Call" else 0
                bet_raise_other_0 = 0 if call_check_other_0 == 1 else bet_size_other

            stacksize_self = starting_stack_self - bet_size_self
            stacksize_other = starting_stack_other - bet_size_other
            pot_size = bet_size_self + bet_size_other
            call_check_self = 0
            bet_raise_self = 0
            call_check_other_1 = 0
            bet_raise_other_1 = 0
            flop = np.zeros(52, dtype=int)
            turn = np.zeros(52, dtype=int)
            river = np.zeros(52, dtype=int)
            cards = np.concatenate([cards, flop, turn, river])
            vector = np.array(
                [
                    position,
                    starting_stack_self,
                    starting_stack_other,
                    street,
                    stacksize_self,
                    stacksize_other,
                    bet_size_self,
                    bet_size_other,
                    pot_size,
                    call_check_self,
                    bet_raise_self,
                    call_check_other_0,
                    bet_raise_other_0,
                    call_check_other_1,
                    bet_raise_other_1,
                ]
            )
            state = np.concatenate([cards, vector])
        else:
            state = previous_state
            street_previous = state[-12]
            stacksize_self = previous_state[-11]
            stacksize_other = previous_state[-10]
            bet_size_self = previous_state[-9]
            bet_size_other = previous_state[-8]
            pot_size = state[-7]
            call_check_self = 0
            bet_raise_self = 0
            call_check_other_0 = 0
            bet_raise_other_0 = 0
            call_check_other_1 = 0
            bet_raise_other_1 = 0
            current_street = street_previous
            last_street_other = street_previous
            # iterate backwards through hand history until the last action by hero is reached
            relevant_actions = []
            finished = False
            for street in ["River", "Turn", "Flop", "PreFlop"]:
                if finished:
                    break
                street_actions = hand_history["Actions"][street]
                if (
                        (street in ("River", "Turn") and street_actions["Card"] is None) or
                        (street == "Flop" and street_actions["Cards"] == [])
                ):
                    continue

                if STREET_DICT[street] > street_previous:
                    current_street = street_previous + 1
                    cards = street_actions["Cards"] if street == "Flop" else [street_actions["Card"]]
                    board_encoded = self.encode_street(cards)
                    if street == "Flop":
                        state[54: 106] = board_encoded
                    elif street == "Turn":
                        state[106: 158] = board_encoded
                    else:
                        state[158: 210] = board_encoded
                if street == "PreFlop":
                    actions = street_actions
                else:
                    actions = street_actions["Actions"]


                if len(actions):
                    actions = copy.deepcopy(actions)
                    while len(actions):
                        action = actions.pop()
                        action["Street"] = street
                        relevant_actions.append(action)
                        if action["Player"] == player_id_self:
                            finished = True
                            break

            for action in reversed(relevant_actions):

                if action["Player"] == player_id_other:
                    if STREET_DICT[action["Street"]] != last_street_other:
                        bet_size_other = 0
                        last_street_other += 1
                    if action["Action"] == "Check":
                        if call_check_other_0 == 0 and bet_raise_other_0 == 0:
                            call_check_other_0 = 1
                        else:
                            call_check_other_1 = 1
                    elif action["Action"] == "Call":
                        if call_check_other_0 == 0 and bet_raise_other_0 == 0:
                            call_check_other_0 = 1
                        else:
                            call_check_other_1 = 1
                        stacksize_other -= action["Amount"]
                        pot_size += action["Amount"]
                        bet_size_other += action["Amount"]
                    else:
                        if call_check_other_0 == 0 and bet_raise_other_0 == 0:
                            bet_raise_other_0 = action["Amount"]
                            add_other = bet_raise_other_0 - bet_size_other

                        else:
                            bet_raise_other_1 = action["Amount"]
                            add_other = bet_raise_other_1 - bet_size_other
                        stacksize_other -= add_other
                        pot_size += add_other
                        bet_size_other += add_other

                else:
                    if current_street != last_street_other:
                        bet_size_other = 0
                    if action["Action"] == "Check":
                        call_check_self = 1
                    elif action["Action"] == "Call":
                        call_check_self = 1
                        stacksize_self -= action["Amount"]
                        pot_size += action["Amount"]
                        bet_size_self += action["Amount"]
                    else:
                        bet_raise_self = action["Amount"]
                        add = bet_raise_self - bet_size_self
                        stacksize_self -= add
                        pot_size += add
                        bet_size_self += add

                    if current_street > street_previous:
                        bet_size_self = 0

            state[-12:] = [
                current_street,
                stacksize_self,
                stacksize_other,
                bet_size_self,
                bet_size_other,
                pot_size,
                call_check_self,
                bet_raise_self,
                call_check_other_0,
                bet_raise_other_0,
                call_check_other_1,
                bet_raise_other_1,
            ]
        return state
    def choose_action(
            self,
            state,
            legal_actions,
            minbet,
            stack_left,
            bet_thus_far,
    ):
        state = state.reshape(1, -1)
        action_probs = self.actor(state)
        logits = tf.math.log(action_probs)
        action = tf.random.categorical(logits, 1)[0, 0].numpy()

        while action not in legal_actions:
            self.add_observation_to_q_buffer(state, action, -2)
            action_probs = self.actor(state)
            logits = tf.math.log(action_probs)
            action = tf.random.categorical(logits, 1)[0, 0].numpy()
        if action == 2:
            alpha, beta = self.betsize_network(state)[0]
            bet_frac = np.random.beta(alpha, beta)

        else:
            bet_frac = None

        bet_size = int(round((minbet + bet_frac * (stack_left + bet_thus_far - minbet)), 1)) if action == 2 else None
        return action, bet_size, bet_frac
