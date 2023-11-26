import tensorflow as tf
from tensorflow_probability import distributions
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
from itertools import product
from Agents.networks import critic_network, actor_network
import tensorflow as tf
import logging

# Set the threshold for what messages will be logged
logging.getLogger('tensorflow').setLevel(logging.ERROR)


# Define the total number of card values and suits
NUM_CARD_VALUES = 13  # From 2 to Ace
NUM_SUITS = 4  # Clubs, Diamonds, Hearts, Spades


class Agent:
    def __init__(self):
        self.actor = actor_network(input_shape=(2,), shared_layers=(512, 256),
                       categorical_branch_layers=(256, 128, 64), continuous_branch_layers=(128, 64),
                       dropout_rate=0.2, num_classes=3)
        self.critic = critic_network()
        self.optimizer = Adam(learning_rate=0.00001)
        self.episode = None
        self.reset_episode()

    def reset_episode(self):
        self.episode = {"inputs": [], "actions": [], "bet_size_fractions": []}
    def add_to_episode(self, inputs, action, bet_size_frac):
        self.episode["inputs"].append(inputs)
        self.episode["actions"].append(action)
        if bet_size_frac is None:
            bet_size_frac = 0
        self.episode["bet_size_fractions"].append(bet_size_frac)

    def custom_beta_loss(self, alpha, beta, chosen_bet_sizes):
        """
        Calculate the loss for bet sizing based on the beta distribution's PDF.
        Since calculating the exact PDF for beta distribution in TensorFlow is complex,
        this function uses a simplified approach that encourages the model's alpha and beta
        outputs to move towards values that increase the density of the chosen bet sizes.
        epsilon is a small number to prevent division by zero or log of zero.
        """
        # Calculate the mean of the beta distribution
        dist = distributions.Beta(alpha, beta)
        log_probs = dist.log_prob(np.squeeze(chosen_bet_sizes+1e-10))  # masked values are 0 and we want to prevent inf
        # Use a simple squared difference as a placeholder for the actual PDF-based loss
        return log_probs


    # Then you use this function in your train_step like so:
    def train_step(self, inputs, actions, reward, chosen_bet_sizes, bet_size_masks):
        #train critic
        expected_rewards = self.critic(inputs)
        reward_tensor = tf.fill(tf.shape(expected_rewards), tf.cast(reward, expected_rewards.dtype))
        self.critic.train_on_batch(inputs, reward_tensor)

        delta_rewards = reward_tensor - expected_rewards
        actions_tensor = tf.constant(actions)
        indices = tf.range(0, len(actions))
        idx_flattened = tf.stack([indices, actions_tensor], axis=1)
        with tf.GradientTape() as tape:
            # Forward pass: get the model's predictions
            action_probs, alpha_outputs, beta_outputs = self.actor(inputs, training=True)
            alpha_outputs = tf.squeeze(alpha_outputs)  # Remove dimensions of size 1
            beta_outputs = tf.squeeze(beta_outputs)  # Remove dimensions of size 1


            # Calculate the bet size loss, masked to apply only for samples with a bet size decision
            #if 1 in bet_size_masks:
            #    bet_size_loss = self.custom_beta_loss(alpha_outputs, beta_outputs, chosen_bet_sizes)
            #    bet_size_loss = tf.reduce_mean(bet_size_loss * bet_size_masks)
            #else:

            bet_size_loss = 0
            # Combine the losses
            action_log_probs = tf.math.log(tf.gather_nd(action_probs, idx_flattened))
            action_loss = -tf.reduce_mean(action_log_probs * delta_rewards)
            combined_loss = action_loss + bet_size_loss


        # Calculate gradients
        grads = tape.gradient(action_loss, self.actor.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        return action_loss
    def train_on_episode(self, reward):
        n_steps = len(self.episode["actions"])
        states = self.episode["inputs"][0]
        hand = self._one_hot_encode_hand(*states[0])
        position = states[1]
        observation = np.tile(np.append(hand, position), (n_steps, 1))
        actions = self.episode["actions"]
        bet_sizes = np.array(self.episode["bet_size_fractions"]).reshape((n_steps, 1))
        bets_mask = np.squeeze(np.where(actions == 2, 1.0, 0.0))
        self.train_step(observation, actions, reward, bet_sizes, bets_mask)
        self.reset_episode()

    def save_model(self, name_actor = "actor_model", name_critic = "critic_model"):
        self.actor.save("saved_models/"+ name_actor)
        self.critic.save("saved_models/" + name_critic)
    def _one_hot_encode_hand(self, card1, card2):
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
    def transform_hand_history(self, hand_history):


        vector = [

        ]
    def choose_action(self, holecards, hand_history, legal_actions, minbet, stack_left, bet_thus_far):
        hand = self._one_hot_encode_hand(*holecards)
        observation = np.append(hand, hand_history)
        state = observation.reshape(1, -1)
        action_probs, alpha, beta = self.actor(state)
        action_probs = action_probs.numpy()[0]
        action = np.random.choice(a=len(action_probs), p=action_probs)
        while action not in legal_actions:
            self.train_step(state, [action], [-2], [0], [0])
            action_probs, alpha, beta = self.actor(state)
            action_probs = action_probs.numpy()[0]
            action = np.random.choice(a=len(action_probs), p=action_probs)

        bet_frac = np.random.beta(alpha[0, 0], beta[0, 0]) if action == 2 else None

        bet_size = int(round((minbet + bet_frac * (stack_left + bet_thus_far - minbet)), 1)) if action == 2 else None
        return action, bet_size, bet_frac
