from pokerAI.modules.split_gru_action import SplitGRUActionModule
from pokerAI.value_functions.split_gru_value_function import GRUValueFunction
from pokerAI.encoders.embedding_hand_encoder import Encoder, feature_dim
from torch import nn
import torch
import numpy as np
from itertools import combinations
from PokerGame.NLHoldem import Game, int_to_card
import numpy as np
import warnings

# Convert all warnings to errors
warnings.filterwarnings("error", category=RuntimeWarning, message="invalid value encountered in multiply")



class SplitGruPolicy(torch.nn.Module):

    all_holecards = np.array(list(combinations(range(52), 2)))
    all_holecards_tuple = np.array([[int_to_card[hole[0]], int_to_card[hole[1]]] for hole in all_holecards])
    encoder = Encoder()
    all_holecards_enc = encoder.encode(all_holecards)

    # generate matrix of valid comparisons

    valid_comparisons = (all_holecards[:, None, :, None]  != all_holecards[None, :, None, :]).all(axis=(2,3))


    def __init__(
            self,
            input_size_recurrent,
            input_size_regular,
            hidden_size,
            num_gru_layers,
            linear_layers,
            num_peaks=3,
            activation=nn.GELU,
            value_function=None,
            idx_stack_start=62,
            idx_stack_now=30,
            idx_bet=41,
    ):
        super().__init__()
        self.module = SplitGRUActionModule(

            input_size_recurrent=input_size_recurrent,
            input_size_regular=input_size_regular,
            hidden_size=hidden_size,
            num_gru_layers=num_gru_layers,
            linear_layers=linear_layers,
            num_peaks=3,
            activation=nn.GELU,
        )

        self.hidden = None
        self.sequence_buffer = None
        self.sequence_buffer_range = None
        if value_function is not None:
            self.value_function = value_function
        else:
            self.value_function = GRUValueFunction(
                input_size_recurrent,
                input_size_regular - 2,
                hidden_size,
                num_gru_layers,
                linear_layers,
                activation,
                output_dim=2,
                idx_stack_start=idx_stack_start,
                idx_stack_now=idx_stack_now,
                idx_bet=idx_bet,
            )

        # Generate all possible hole card combinations
        # Initialize probabilities
        self.starting_range = np.full(len(self.all_holecards), 1.0 / 1326)
        self.reset()


    def reset(self):
        self.hidden = torch.zeros(self.module.split_gru.gru.num_layers, 1, self.module.split_gru.gru.hidden_size)
        self.hidden_range = torch.zeros(self.module.split_gru.gru.num_layers, 1326, self.module.split_gru.gru.hidden_size)
        if self.value_function:
            self.hidden_value = torch.zeros(self.value_function.gru.num_layers, 1, self.value_function.gru.hidden_size)
            self.hidden_value_range = torch.zeros(self.value_function.gru.num_layers, 1326, self.value_function.gru.hidden_size)
        self.sequence_buffer = []
        self.sequence_buffer_range = []
        self.range = self.starting_range
        self.all_cards_enc = self.all_holecards_enc.clone()

    def encode_state(self, state, range=True):
        """encodes state into recurrent releveant and static"""
        n_players_int = state["n_players"]
        n_players = np.zeros(9)  # min 2 max 10 players
        n_players[n_players_int - 2] = 1
        starting_stacks = np.zeros(10)
        starting_stacks[: n_players_int] = state["starting_stacks"]
        street = np.zeros_like(starting_stacks)
        street[state["street"]] = 1
        stacks = np.full(10, -1) # stack of 0 is a meaningful value, thus -1 as none
        stacks[: n_players_int] = state["stacks"]
        players_left = np.zeros_like(starting_stacks)
        players_left[: n_players_int] = state["in_play"]
        position = np.zeros_like(starting_stacks)
        position[state["position"]] = 1
        bets = np.zeros_like(position)
        bets[: n_players_int] = state["bets"]

        recurrent_state = torch.tensor(
            np.hstack([
                stacks,
                players_left,
                street,
                state["stack"],
                bets,
                state["bet"],
                state["pot"],
            ]), dtype=torch.float32
        )

        static_state = torch.tensor(
            np.hstack([
                n_players,
                starting_stacks,
                state["starting_stack"],
                position
            ]), dtype=torch.float32
        )

        if range:
            game_state = torch.cat([recurrent_state, static_state], dim=-1)
            features = torch.cat([
                game_state.repeat(1326, 1),
                self.all_cards_enc
            ], dim=-1)
        else:
            hole_int = [card.int for card in state["holecards"]]
            if len(state["board"]) > 0:
                board_int = [card.int for card in state["board"]]
                flop = board_int[:3]
                if len(board_int) > 3:
                    turn = board_int[3]
                    if len(board_int) == 5:
                        river = board_int[4]
                    else:
                        river = None
                else:
                    turn, river = None, None
            else:
                flop, turn, river = None, None, None
            card_enc = self.encoder.encode(hole_int, flop, turn, river)
            features = torch.cat([
                recurrent_state,
                static_state,
                card_enc.squeeze(),
            ], dim=-1)
        return features

    def remove_cards_from_range(self, cards):
        # Find combinations that include any of the dealt cards
        invalid_combinations = np.any(np.isin(self.all_holecards, cards), axis=1)

        # Set probabilities of invalid combinations to 0
        self.range[invalid_combinations] = 0

        # Renormalize the probabilities
        total_prob = np.sum(self.range)
        self.range /= total_prob

    def bayesian_update(self, action_probs, betfrac_density=None):
        """
        Perform a Bayesian update on the probabilities of hole card combinations.

        Args:
            prior_probs (np.ndarray): Prior probabilities of each hole card combination (shape: (1326,)).
            action_probs (np.ndarray): Probabilities of performing an action with each hole card combination (shape: (1326,)).

        Returns:
            np.ndarray: Updated (posterior) probabilities (shape: (1326,)).
        """
        # Ensure inputs are NumPy arrays
        action_probs = action_probs

        # Compute the unnormalized posterior probabilities
        if betfrac_density is None:
            betfrac_density = 1
        try:
            unnormalized_posterior = self.range * action_probs * np.clip(betfrac_density, None, 50)


        # Normalize the posterior probabilities
            self.range = unnormalized_posterior / np.sum(unnormalized_posterior)
        except:
            print("something crazy is happening")
            l = "lalalala"
            print("insane")


    def update_feature_array(self, flop, turn, river):
        flop_array = np.tile(flop, (1326, 1)) if flop is not None else None
        turn_array = np.tile(turn, (1326, 1)) if turn is not None else None
        river_array = np.tile(river, (1326, 1)) if river is not None else None
        self.all_cards_enc = self.encoder.encode(self.all_holecards, flop_array, turn_array, river_array)


    def create_sequence(self, range=False):
        """generate 3-d tensor array to pass through GRU from states."""
        if range:
            state = torch.stack(self.sequence_buffer, dim=1)
        else:
            state = torch.vstack(self.sequence_buffer).unsqueeze(0)
        if self.value_function:
            if range:
                value_prediction, self.hidden_value_range = self.value_function(state,
                                                                                hidden_state=self.hidden_value_range,
                                                                                return_sequences=True)
            else:
                value_prediction, self.hidden_value = self.value_function(state, hidden_state=self.hidden_value, return_sequences=True)
            state = torch.cat([state, value_prediction], dim=-1)
        return state

    def create_sequence_range(self):
        """generate 3-d tensor array to pass through GRU from states."""
        state = torch.stack(self.sequence_buffer_range, dim=1)
        if self.value_function:
            value_prediction, self.hidden_value_range = self.value_function(state, hidden_state=self.hidden_value_range, return_sequences=True)
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
    def get_action(
            self,
            state,
            smallest_unit=1,
            temperature=1,
            update_range=True,
            play_range=False,
    ):
        # Retrieve specific arguments like `game` from kwargs
        sequence = self.create_sequence(play_range)
        legal_actions = list(state["legal_actions"])
        valid_action_mask = torch.zeros(3, dtype=torch.bool)  # Initialize as all False
        valid_action_mask[legal_actions] = True
        valid_action_mask = valid_action_mask.reshape(1,1, -1)
        h = self.hidden_range if play_range else self.hidden
        dist, h_act = self(sequence, h, legal_actions_mask=valid_action_mask)
        if update_range and not play_range:
            sequence_range = self.create_sequence_range()
            dist_range, self.hidden_range = self(sequence_range, self.hidden_range, legal_actions_mask=valid_action_mask)
        else:
            dist_range = dist
        self.hidden = h_act
        if play_range:
            self.hidden_range = h_act
        min_bet = state["legal_betsize"]
        max_bet = state["bet"] + state["stack"]

        # sample hand from range
        try:
            action_type, betfrac = dist.sample(temperature=temperature)
        except:
            k = 1
            print(k)
            dist, h_act = self(sequence, h, legal_actions_mask=valid_action_mask)
            action_type, betfrac = dist.sample(temperature=temperature)
        if play_range:
            hand_ind = np.random.choice(1326, p=self.range)
            action_type = action_type[hand_ind].item()
            betfrac = betfrac[hand_ind].item()
        else:
            action_type = action_type.item()
            betfrac = betfrac.item()
        if action_type in (2, 3, 4):
            action = 2

        else:
            action = action_type
            betfrac = None


        if action_type == 2:
            stack = state["stacks"][state["position"]]
            spr = stack / state["pot"]
            # make betsize depend on spr
            betfrac_spr = betfrac ** (spr**0.5)
            # create 5% mass for minbet and allin

            if betfrac_spr < 0.05:
                betfrac_transformed = 0
            elif betfrac_spr > 0.95:
                betfrac_transformed = 1
            else:
                betfrac_transformed = (betfrac_spr - 0.05)/0.9
            betsize = (max_bet - min_bet) * betfrac_transformed + min_bet
            betsize = round(betsize / smallest_unit) * smallest_unit
            betsize = min(max_bet, betsize)
        else:
            betsize = None
        # Return action and any additional info (if needed)

        if betfrac is not None:
            betfrac_tensor = torch.full((1326,1), betfrac, dtype=torch.float32)
            if play_range:
                action_tensor = torch.full((1326,1), action, dtype=torch.float32)
                probs = torch.exp(
                    dist_range.log_prob(action_tensor, betfrac_tensor.to(torch.double)).squeeze()).to(torch.float)
                cat_probs = probs[:, 0].numpy()
                bet_density = probs[:, 1].numpy()
        else:
            cat_probs = dist_range.category_probs[..., action_type].squeeze().numpy()
            bet_density = None
        if update_range:
            self.bayesian_update(cat_probs, bet_density)
        self.sequence_buffer = []
        self.sequence_buffer_range = []

        return action, betsize, action_type, betfrac, legal_actions


if __name__ == "__main__":

    hidden_size = 128
    input_size_recurrent = 42
    input_size_static = 30
    linear_layers = (256, 256)
    policy = SplitGruPolicy(
        input_size_recurrent,
        feature_dim + input_size_static +2 +3,
        hidden_size,
        1,
        linear_layers,
        value_function=GRUValueFunction(
            input_size_recurrent,
            feature_dim + input_size_static + 3,
            hidden_size,
            1,
            linear_layers,
        )
    )
    game = Game()
    game.new_hand()
    acting_player = game.acting_player
    state = game.get_state(acting_player)
    recurrent_state, static_state = policy.encode_state(state)
    # update range
    policy.remove_cards_from_range(acting_player.hole_int)
    preflop_features = policy.encoder.encode(acting_player.hole_int)
    features = torch.cat([
        recurrent_state,
        static_state,
        preflop_features.squeeze(),
    ], dim=-1)
    policy.add_to_sequence(features)
    action, betsize, action_type, betfrac, legal_actions = policy.get_action(state)
    game.implement_action(acting_player, action, betsize)

