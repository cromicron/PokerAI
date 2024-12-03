from encoders.state_encoder import encode_state
from encoders.strength_encoder import encode_strength
import torch
import numpy as np

class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.legal_actions = []
        self.reward = None

    def add_observation(self, observation):
        self.observations.append(observation)

    def add_action(self, action, legal_actions):
        self.actions.append(action)
        self.legal_actions.append(action)

    def finish_episode(self, reward):
        self.reward = reward



class Agent:
    def __init__(self, player, policy, policy_path=None):
        self.player = player
        self.policy = policy
        if policy_path:
            self.policy.load_state_dict(torch.load(policy_path))
        self.episodes = []
        self.create_episode()

    def get_action(self, game, smallest_unit=1):
        return self.policy.get_action(game, smallest_unit)

    def add_to_sequence(self, state):
        self.policy.add_to_sequence(state)

    def clear_episode_buffer(self):
        del self.episodes
        self.episodes = []

    def encode_state(self, game):
        game_state = encode_state(game, self.player)
        hand_strength = encode_strength(self.player.holecards, game.board)
        return np.hstack([game_state, hand_strength])

    def create_episode(self):
        self.episodes.append(Episode())

