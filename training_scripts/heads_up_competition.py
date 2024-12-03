from encoders.state_encoder import encode_state
from encoders.strength_encoder import encode_strength
from PokerGame.NLHoldem import Game, Card, Player
from value_functions.gru_value_function import GRUValueFunction
from policies.mixed_gru_policy import MixedGruPolicy
from itertools import combinations
import csv
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import copy
import gc
from algos.ppo import Agent, Episode, PPOTrainingDataset
from itertools import combinations

num_agents = 8
hands_per_pair = 400
players_per_hand = 2
stack = 20

game = Game()
game.new_hand(first_hand=True)

# Compute state vector size using a dummy state
dummy_player = game.players[0]
dummy_state = encode_state(game, dummy_player)
dummy_cards = dummy_player.holecards
dummy_strength = encode_strength(dummy_cards)
state_vector_value = torch.tensor(
    np.hstack([dummy_state, dummy_strength]), dtype=torch.float32
)
state_vector_value = state_vector_value.shape[0]

agents = {
    Player(i, stack): MixedGruPolicy(
        input_size=state_vector_value+3,
        hidden_size=256,
        num_gru_layers=1,
        linear_layers=(256, 128),
        value_function= GRUValueFunction(
            input_size=state_vector_value,
            hidden_size=256,
            num_gru_layers=1,
            linear_layers=(256, 128),
            output_dim=3,
        )

    ) for i in range(num_agents)
}
pairs = list(combinations(agents.keys(), 2))

for pair in pairs:
    game.players=list(pairs)
