from modules.mixed_gru_module import MixedGRUModule
from torch import nn
import torch
import numpy as np


class MixedGruPolicy(torch.nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size=128,
            num_gru_layers=1,
            linear_layers=(128, 128),
            num_peaks=3,
            activation=nn.LeakyReLU,
            single_raise=True,
            value_function=None,
    ):
        super().__init__()
        self.module = MixedGRUModule(
            input_size,
            hidden_size,
            num_gru_layers,
            linear_layers,
            num_peaks,
            activation,
            single_raise,
        )
        self.single_raise=single_raise
        self.hidden = None
        self.sequence_buffer = None
        self.value_function = value_function
        self.reset()



    def reset(self):
        self.hidden = torch.zeros(self.module.gru.gru.num_layers, 1, self.module.gru.gru.hidden_size)
        if self.value_function:
            self.hidden_value = torch.zeros(self.value_function.gru.num_layers, 1, self.value_function.gru.hidden_size)
        self.sequence_buffer = []
    def create_sequence(self):
        """generate 3-d tensor array to pass through GRU from states."""
        state = torch.tensor(
            np.stack(self.sequence_buffer), dtype=torch.float32
        ).unsqueeze(0)
        if self.value_function:
            value_prediction, self.hidden_value = self.value_function(state, hidden_state=self.hidden_value, return_sequences=True)
            state = torch.cat([state, value_prediction], dim=-1)
        return state

    def add_to_sequence(self, state):
        """
        adds observation to current episode buffer. All env steps till action
        must be preserved and passed through the GRU.
        """
        self.sequence_buffer.append(state)

    def forward(self, x, hidden_state=None, return_sequences=False, legal_actions_mask=None):
        return self.module.forward(x, hidden_state, return_sequences, legal_actions_mask)

    @torch.no_grad()
    def get_action(self, game, smallest_unit=1, temperature=1):
        # Retrieve specific arguments like `game` from kwargs
        sequence = self.create_sequence()
        legal_actions = list(game.get_legal_actions())
        valid_action_mask = torch.zeros(3, dtype=torch.bool)  # Initialize as all False
        valid_action_mask[legal_actions] = True
        valid_action_mask = valid_action_mask.reshape(1,1, -1)
        dist, h_act = self(sequence, self.hidden, legal_actions_mask=valid_action_mask)
        self.hidden = h_act
        min_bet = game.get_legal_betsize()
        max_bet = game.acting_player.bet + game.acting_player.stack


        action_type, betfrac = dist.sample(temperature=temperature)
        action_type = action_type.item()
        betfrac = betfrac.item()
        if action_type in (2, 3, 4):
            action = 2
            if not self.single_raise and (action_type != 4):
                betfrac = None
        else:
            action = action_type
            betfrac = None


        if not self.single_raise:
            if action_type == 2:
                betsize = min_bet
                betfrac = None
            elif action_type == 3:
                betsize = max_bet
                betfrac = None
            elif action_type == 4:
                betsize = (max_bet - min_bet) * betfrac + min_bet
                betsize = round(betsize / smallest_unit) * smallest_unit
                betsize = min(max_bet, betsize)
            else:
                betsize = None
        else:
            if action_type == 2:
                # create 5% mass for minbet and allin
                if betfrac < 0.05:
                    betfrac_transformed = 0
                elif betfrac > 0.95:
                    betfrac_transformed = 1
                else:
                    betfrac_transformed = (betfrac - 0.05)/0.9
                betsize = (max_bet - min_bet) * betfrac_transformed + min_bet
                betsize = round(betsize / smallest_unit) * smallest_unit
                betsize = min(max_bet, betsize)
            else:
                betsize = None
        # Return action and any additional info (if needed)

        self.sequence_buffer = []

        return action, betsize, action_type, betfrac, legal_actions
