import tensorflow as tf
from tensorflow_probability import distributions
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
from itertools import product


# Define the total number of card values and suits
NUM_CARD_VALUES = 13  # From 2 to Ace
NUM_SUITS = 4  # Clubs, Diamonds, Hearts, Spades


class Agent:
    def __init__(self):
        self.model = None
        self.optimizer = Adam(learning_rate=0.0001)
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


    def create_network(self, input_shape=(52,), shared_layers=(512, 256),
                       categorical_branch_layers=(128, 64), continuous_branch_layers=(128, 64),
                       dropout_rate=0.2, num_classes=3):
        # Input layer
        inputs = Input(shape=input_shape)

        # Shared layers
        x = inputs
        for num_neurons in shared_layers:
            x = Dense(num_neurons)(x)
            x = LeakyReLU()(x)
            x = Dropout(dropout_rate)(x)

        # Categorical branch for choosing action type
        categorical_x = x
        for num_neurons in categorical_branch_layers:
            categorical_x = Dense(num_neurons)(categorical_x)
            categorical_x = LeakyReLU()(categorical_x)
            categorical_x = Dropout(dropout_rate)(categorical_x)
        categorical_output = Dense(num_classes, activation='softmax', name='categorical_output')(categorical_x)

        # Continuous branch for bet sizing - outputs alpha and beta parameters for the beta distribution
        continuous_x = x
        for num_neurons in continuous_branch_layers:
            continuous_x = Dense(num_neurons)(continuous_x)
            continuous_x = LeakyReLU()(continuous_x)
            continuous_x = Dropout(dropout_rate)(continuous_x)
        # Output the alpha and beta as softplus to ensure they are positive and suitable for beta distribution
        alpha_output = Dense(1, activation='softplus', name='alpha_output')(continuous_x)
        beta_output = Dense(1, activation='softplus', name='beta_output')(continuous_x)

        # Create model
        model = Model(inputs=inputs, outputs=[categorical_output, alpha_output, beta_output])

        # Compile the model with Adam optimizer and custom loss function
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss={'categorical_output': 'categorical_crossentropy',
                            'alpha_output': 'mse',
                            'beta_output': 'mse'},
                      metrics={'categorical_output': 'accuracy'})

        self.model = model

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
    #@tf.function
    def train_step(self, inputs, actions, reward, chosen_bet_sizes, bet_size_masks):
        actions_tensor = tf.constant(actions)
        indices = tf.range(0, len(actions))
        idx_flattened = tf.stack([indices, actions_tensor], axis=1)
        with tf.GradientTape() as tape:
            # Forward pass: get the model's predictions
            action_probs, alpha_outputs, beta_outputs = self.model(inputs, training=True)
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
            action_loss = -tf.reduce_mean(action_log_probs) * reward
            combined_loss = action_loss + bet_size_loss


        # Calculate gradients
        grads = tape.gradient(action_loss, self.model.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

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

    def save_model(self, name = "policy_gradient_model"):
        self.model.save("saved_models/"+ name)
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
        action_probs, alpha, beta = self.model(state)
        action_probs = action_probs.numpy()[0]
        action = np.random.choice(a=len(action_probs), p=action_probs)
        while action not in legal_actions:
            self.train_step([state], [action], [-20], [0], [0])
            action_probs, alpha, beta = self.model(state)
            action_probs = action_probs.numpy()[0]
            action = np.random.choice(a=len(action_probs), p=action_probs)

        bet_frac = np.random.beta(alpha[0, 0], beta[0, 0]) if action == 2 else None

        bet_size = int(round((minbet + bet_frac * (stack_left + bet_thus_far - minbet)), 1)) if action == 2 else None
        return action, bet_size, bet_frac
