import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from Agents.networks import q_network_actor, actor_with_regret
import logging

# Set the threshold for what messages will be logged
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# Define the total number of card values and suits
NUM_CARD_VALUES = 13  # From 2 to Ace
NUM_SUITS = 4  # Clubs, Diamonds, Hearts, Spades


class EpisodeBuffer:
    def __init__(self, input_size: int, output_size: int, buffer_size: int = 1000):
        self.observations = np.empty((0, input_size))
        self.actions = np.empty((0, 1))
        self.evs = np.empty((0, output_size))
        self.buffer_size = buffer_size
class Agent:
    def __init__(self, batch_size = 32):
        self.actor = actor_with_regret()
        self.q_action = q_network_actor()
        self.episode_buffer = EpisodeBuffer(input_size=2, output_size=3)
        self.episodes_q_trained = 0
        self.batch_size = batch_size
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


    def save_model(self, name_actor = "actor_model", name_q = "q_model"):
        self.actor.save("saved_models/"+ name_actor)
        self.q_action.save("saved_models/" + name_q)
    def one_hot_encode_hand(self, card1, card2):
        # Calculate the unique index for each card based on value and suit

        card1_index = (card1.value - 2) * 4 + card1.suit  # Since value is 2-14, subtract 2 to start index at 0
        card2_index = (card2.value - 2) * 4 + card2.suit

        # Create a one-hot encoded vector of size 52 for the deck of cards
        encoded_hand = np.zeros(52, dtype=int)
        encoded_hand[card1_index] = 1
        encoded_hand[card2_index] = 1
        strength = 1 if card1.value == card2.value else 0
        add_strength = 0.01*card1.value if card1.value == card2.value else 0.01* max(card1.value, card2.value)+ 0.0001*min(card1.value, card2.value)
        encoded_hand = np.array([strength + add_strength])
        return encoded_hand

    def choose_action(
            self,
            holecards,
            hand_history,
            legal_actions,
    ):
        hand = self.one_hot_encode_hand(*holecards)
        observation = np.append(hand, hand_history)
        state = observation.reshape(1, -1)
        action_probs = self.actor(state)
        logits = tf.math.log(action_probs)


        action = tf.random.categorical(logits, 1)[0, 0].numpy()
        while action not in legal_actions:
            self.add_observation_to_q_buffer(state, action, -2)
            action_probs = self.actor(state)
            logits = tf.math.log(action_probs)
            action = tf.random.categorical(logits, 1)[0, 0].numpy()

        return action