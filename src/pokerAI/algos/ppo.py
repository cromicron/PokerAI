from pokerAI.encoders.state_encoder import encode_state
from pokerAI.encoders.strength_encoder import encode_strength
from PokerGame.NLHoldem import Game, Card
from pokerAI.value_functions.gru_value_function import GRUValueFunction
from pokerAI.policies.mixed_gru_policy import MixedGruPolicy
from itertools import combinations
import csv
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import copy
import gc
import time
import ray
from pokerAI.modules.mixed_distribution_head import stop_execution


torch.set_flush_denormal(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_tensor(name):
    def hook(grad):
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print(f"NAN Gradient issue in {name}")
        if abs(grad).any() > 10:
            print(f"Large Gradient issue in {name}")
        print(f"max grad {torch.max(grad)}")
    return hook
class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.legal_actions = []
        self.reward = None
        self.ranges = []

    def add_observation(self, observation):
        self.observations.append(observation)

    def add_action(self, action, legal_action):
        self.actions.append(action)
        self.legal_actions.append(legal_action)

    def finish_episode(self, reward):
        self.reward = reward

    def add_range(self, hand_range):
        self.ranges.append(hand_range)


def create_training_data(episodes, single_raise=True, value_function=None, ranges=False):
    """
    Processes a list of poker hand episodes into time-series tensors for GRU training.
    Dynamically determines the max episode length and pads accordingly.

    Args:
        episodes (list of Episode): List of episodes containing observations, actions, and rewards.
        single raise (bool): whether bet/raise is encoded by only one value

    Returns:
        tuple: A tuple containing:
            - observations (torch.Tensor): (batch_size, max_steps, observation_dim)
            - action_types (torch.Tensor): (batch_size, max_steps) [Categorical]
            - betfracs (torch.Tensor): (batch_size, max_steps) [Optional betfrac]
            - masks (torch.Tensor): (batch_size, max_steps) [Valid timestep mask]
            - action_masks (torch.Tensor): (batch_size, max_steps) [Valid action mask]
            - rewards (torch.Tensor): (batch_size,)
    """
    # Determine the maximum number of timesteps (longest sequence)
    max_steps = max(len(episode.observations) for episode in episodes)
    batch_size = len(episodes)
    obs_dim = episodes[0].observations[0].shape[1] if ranges else len(episodes[0].observations[0])
    index_cont_raise = 2 if single_raise else 4
    # Initialize padded tensors
    if ranges:
        batch_size *= 1326
        ranges_tensor = torch.zeros((batch_size, max_steps), dtype=torch.float32)
    observations = torch.zeros((batch_size, max_steps, obs_dim), dtype=torch.float32)
    action_types = torch.zeros((batch_size, max_steps), dtype=torch.long)  # Categorical
    betfracs = torch.zeros((batch_size, max_steps), dtype=torch.float32)  # Optional betfrac
    masks = torch.zeros((batch_size, max_steps), dtype=torch.bool)  # Valid timestep mask
    action_masks = torch.zeros((batch_size, max_steps), dtype=torch.bool)  # Valid action mask
    legal_actions_mask = torch.zeros((batch_size, max_steps, index_cont_raise+1), dtype=torch.bool)
    rewards = torch.zeros((batch_size,), dtype=torch.float32)

    for i, episode in enumerate(episodes):
        num_steps = len(episode.observations)
        if ranges:
            i_start = i*1326
            i_end = i_start + 1326
            # Add observations
            observations[i_start:i_end, :num_steps, :] = torch.stack(episode.observations, dim=1)
            ranges_tensor[i_start:i_end, :num_steps] = torch.tensor(
                np.stack(episode.ranges).T, dtype=ranges_tensor.dtype)
            # Add action types and betfracs
            for j, action in enumerate(episode.actions):
                if action is not None:  # Only process valid actions
                    action_type, betfrac = action  # Deconstruct action tuple
                    action_types[i_start:i_end, j] = action_type
                    action_masks[i_start:i_end, j] = True  # Mark this timestep as having a valid action
                    if action_type == index_cont_raise:  # Intermediary bet: betfrac is valid
                        betfracs[i_start:i_end, j] = betfrac
                    legal_actions_mask[i_start:i_end, j, episode.legal_actions[j]] = True


            # Mark valid timesteps in the general mask
            masks[i_start:i_end, :num_steps] = True

            # Add rewards (only one per episode, not time-series)
            rewards[i_start:i_end] = torch.tensor(episode.reward, dtype=rewards.dtype)
        else:
            # Add observations
            observations[i, :num_steps, :] = torch.tensor(
                episode.observations, dtype=torch.float32
            )
            # Add action types and betfracs
            for j, action in enumerate(episode.actions):
                if action is not None:  # Only process valid actions
                    action_type, betfrac = action  # Deconstruct action tuple
                    action_types[i, j] = action_type
                    action_masks[i, j] = True  # Mark this timestep as having a valid action
                    if action_type == index_cont_raise:  # Intermediary bet: betfrac is valid
                        betfracs[i, j] = betfrac
                    legal_actions_mask[i, j, episode.legal_actions[j]] = True

        # Mark valid timesteps in the general mask
            masks[i, :num_steps] = True

            # Add rewards (only one per episode, not time-series)
            rewards[i] = episode.reward
    if value_function:
        value_function.to(device)
        with torch.no_grad():
            n_data = observations.size(0)
            batch_size_value = n_data//10
            qs_list = []
            for k in range(0, n_data, batch_size_value):
                observations_batch = observations[k: k+batch_size_value].to(device)
                qs_list.append(value_function(observations_batch, return_sequences=True)[0])
            qs = torch.vstack(qs_list).to("cpu")
            torch.cuda.empty_cache()
            del(qs_list)
            gc.collect()
    else:
        qs = None
    data_vars = ["observations", "action_types", "betfracs", "masks", "action_masks", "rewards", "legal_actions_mask", "qs"]
    if ranges:
        data_vars.append("ranges_tensor")
    return {key: value for key, value in locals().items() if key in data_vars}

class Agent:
    def __init__(
            self,
            player,
            policy,
            value_function,
            encode_value=True,
            idx_stack_start=62,
            idx_stack_now=30,
            idx_bet=41,
            path_policy_network=None,
            path_value_network=None,
            idx_pot=42,
            create_optimizer=True,
    ):
        self.player = player
        self.policy = policy
        self.episodes = []
        self.training_data = None
        self.value_function = value_function
        if create_optimizer:
            self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-4)
            self.value_optimizer = torch.optim.AdamW(self.value_function.parameters(), lr=1e-4)
        else:
            self.policy_optimizer, self.value_optimizer = None, None
        self.encode_value = encode_value
        if self.encode_value:
            self.policy.value_function = value_function
        self.idx_stack_start=idx_stack_start
        self.idx_stack_now=idx_stack_now
        self.idx_bet=idx_bet
        self.idx_pot = idx_pot
        self.blind=0
        if path_policy_network is not None:
            self.policy.load_state_dict(torch.load(path_policy_network, map_location="cpu", weights_only=True))
            self.policy.to("cpu")  # Ensure policy is explicitly on CPU
            self.policy.eval()
        if path_value_network is not None:
            self.value_function.load_state_dict(
                torch.load(path_value_network, map_location="cpu", weights_only=True))

    def encode_state(self, state, range=False):
        return self.policy.encode_state(state, range)

    def get_action(self, game, smallest_unit=1, temperature=1, play_range=False):
        return self.policy.get_action(game, smallest_unit, temperature=temperature, play_range=play_range)

    def add_to_sequence(self, state):
        self.policy.add_to_sequence(state)

    def add_to_sequence_range(self, state):
        self.policy.add_to_sequence_range(state)
    def generate_training_data(self, ranges=True):
        """generate training data from episodes"""
        value_function = self.value_function if self.encode_value else None
        self.training_data = create_training_data(
            self.episodes, value_function=value_function, ranges=ranges,
        )

    def clear_episode_buffer(self):
        del self.episodes
        self.episodes = []
        gc.collect()

    def reset(self):
        self.clear_episode_buffer()
        self.player.stack = self.player.starting_stack
        self.blind=0
        self.training_data = None

    @staticmethod
    def compute_action_weights(ranges_tensor: torch.Tensor,
                               old_log_probs: torch.Tensor,
                               action_types: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized posterior weights for each hand after observing the action and (if action=2) betsize.

        Args:
            ranges_tensor: shape [n * 1326, timesteps], prior probability of holding each hand.
            old_log_probs: shape [n * 1326, timesteps, 2],
                           log_probs[..., 0] = log-prob of chosen action,
                           log_probs[..., 1] = log-prob of betsize (if action = 2).
            action_types: shape [n * 1326, timesteps], each in {0,1,2} — already correctly aligned.

        Returns:
            weights: shape [n * 1326, timesteps] with normalized posterior weights.
        """
        n_hands = 1326

        # Get log-prob of the chosen action for each hand
        log_action_probs = old_log_probs[..., 0]

        # Get log-prob of the betsize for each hand (only used if action == 2)
        log_betsize_probs = old_log_probs[..., 1]

        # Add the betsize log-prob only if action = 2
        log_total_probs = log_action_probs.clone()
        mask_bet = (action_types == 2)
        log_total_probs[mask_bet] += log_betsize_probs[mask_bet]

        # Compute unnormalized weights
        weights = ranges_tensor * torch.exp(log_total_probs)

        # Normalize over 1326 hands for each time step
        weights = weights.view(-1, n_hands, weights.shape[-1])
        weights /= weights.sum(dim=1, keepdim=True)

        # Flatten back to original shape [n * 1326, timesteps]
        weights = weights.view(-1, weights.shape[-1])*1326

        return weights

    def train_policy(
            self,
            epochs=4,
            clip_epsilon_cat=0.2,
            clip_epsilon_cont=0.2,
            entropy_coef_cat=0.01,
            entropy_coef_size=0.01,
            batch_size=64,
            verbose=True,
            ranges=False,
    ):
        """
        Trains the policy using PPO, with `old_log_probs` directly included in the DataLoader batches.
        Includes logging for verbosity.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_epsilon = torch.tensor([clip_epsilon_cat, clip_epsilon_cont], device=device, dtype=torch.float32).unsqueeze(0)
        self.policy.to(device)
        self.policy.eval()
        observations = self.training_data["observations"].to(device)
        with torch.no_grad():
            if self.encode_value:
                qs = self.training_data["qs"].to(device)
                observations = torch.cat([observations, qs], dim=-1)
                n_data = observations.size(0)
                batch_size_data = n_data // 10
                probs_list = []
                for k in range(0, n_data, batch_size_data):
                    observations_batch = observations[k: k + batch_size_data].to(device)
                    probs_list.append(self.policy(observations_batch, return_sequences=True)[0].category_probs)
                probs = torch.vstack(probs_list)
                torch.cuda.empty_cache()
                del (probs_list)
                gc.collect()
                torch.cuda.empty_cache()
                blind = observations[:, 0, self.idx_bet]
                starting_stack = observations[..., self.idx_stack_start]
                current_stack = observations[..., self.idx_stack_now]
                q_fold = (current_stack - starting_stack + blind.unsqueeze(-1)).unsqueeze(-1)
                qs = torch.cat([q_fold, qs], dim=-1)
                predicted_values = (qs * probs).sum(dim=-1)
            else:
                predicted_values, _ = self.value_function(observations, return_sequences=True)
                predicted_values = predicted_values.squeeze(-1)
        action_types = self.training_data["action_types"].to(device)
        betfracs = self.training_data["betfracs"].to(device)
        rewards = self.training_data["rewards"].to(device)
        masks = self.training_data["masks"].to(device)
        action_masks = self.training_data["action_masks"].to(device)
        legal_actions_mask = self.training_data["legal_actions_mask"].to(device)
        if ranges:
            ranges_tensor = self.training_data["ranges_tensor"].to(device)

        with torch.no_grad():
            returns = torch.zeros_like(predicted_values)
            advantages = torch.zeros(*predicted_values.shape, 2).to(device)
            if ranges:
                for i in range(0, rewards.size(0), 1326):
                    last_valid_idx = masks[i].nonzero()[-1].item()
                    returns[i: i+1326, last_valid_idx] = rewards[i:i+1326]
                    for t in reversed(range(last_valid_idx)):
                        returns[i: i+1326, t] = predicted_values[i: i+1326, t + 1] * masks[i: i+1326, t + 1]
                    advantages[i: i+1326, :, 0] = returns[i: i+1326] - predicted_values[i: i+1326]
                    advantages[i: i+1326, :, 1] = returns[i: i+1326] - qs[i: i+1326, :, 2]
            else:
                for i in range(rewards.size(0)):
                    last_valid_idx = masks[i].nonzero()[-1].item()
                    returns[i, last_valid_idx] = rewards[i]
                    for t in reversed(range(last_valid_idx)):
                        returns[i, t] = predicted_values[i, t + 1] * masks[i, t + 1]
                    advantages[i, :, 0] = returns[i] - predicted_values[i]
                    advantages[i, :, 1] = returns[i] - qs[i, :, 2]


        old_policy_module = self.policy.module if hasattr(self.policy, "module") else self.policy
        old_policy = copy.deepcopy(old_policy_module)

        # Precompute old log-probs
        n_data = observations.size(0)
        batch_size_data = n_data // 10
        old_log_probs_list = []
        old_log_probs_fold_list = []
        with torch.no_grad():
            for k in range(0, n_data, batch_size_data):
                observations_batch = observations[k: k + batch_size_data].to(device)
                old_distribution, _ = old_policy(
                    observations_batch, return_sequences=True, action_mask=legal_actions_mask[k: k + batch_size_data])
                action_types_batch = action_types[k: k + batch_size_data]
                betfrac_batch = betfracs[k: k + batch_size_data]
                old_log_probs_batch = old_distribution.log_prob(action_types_batch, value=betfrac_batch)
                old_log_probs_fold_batch = old_distribution.log_prob(
                    torch.zeros_like(action_types_batch), value=torch.zeros_like(betfrac_batch))[..., 0]
                old_log_probs_list.append(old_log_probs_batch)
                old_log_probs_fold_list.append(old_log_probs_fold_batch)

        old_log_probs = torch.vstack(old_log_probs_list)
        old_log_probs_fold = torch.vstack(old_log_probs_fold_list)

        advantages_fold = q_fold.squeeze() - predicted_values

        old_policy.eval()

        # each sample must be weighted, because it isn't equally likely to be part of the dataset if ranges
        with torch.no_grad():
            weight_sample = self.compute_action_weights(ranges_tensor, old_log_probs, action_types)
        weight_sample[~action_masks] = 0
        p_fold_hand = probs[..., 0]
        weight_fold_sample = ranges_tensor * p_fold_hand *1326
        weight_sample = torch.where(action_types==0, 0, weight_sample)
        # to make overall magnitude identical mutliply original weights with 1- p_fold
        weight_sample *= 1- p_fold_hand


        # normalize advantages
        # 1 categorical advantages
        weights_all = torch.vstack([weight_sample, weight_fold_sample])
        advantages_cat = torch.vstack([advantages[...,0], advantages_fold])
        weighted_mean_cat = (advantages_cat * weights_all).sum()/ weights_all.sum()
        weighted_var_cat = ((advantages_cat - weighted_mean_cat) ** 2 * weights_all).sum() / weights_all.sum()
        weighted_std_cat = torch.sqrt(weighted_var_cat + 1e-8)
        advantages_cat = (advantages_cat - weighted_mean_cat)/weighted_std_cat
        advantages_fold = advantages_cat[advantages_cat.size(0)//2:]
        # 2 continuous advantages
        weights_raise = torch.where(action_types==2, weight_sample, 0.0)
        weighted_mean_cont = (advantages[..., 1] * weights_raise).sum()/ weights_raise.sum()
        weighted_var_cont = ((advantages[..., 1] - weighted_mean_cont) ** 2 * weights_raise).sum() / weights_raise.sum()
        weighted_std_cont = torch.sqrt(weighted_var_cont + 1e-8)
        advantages_cont = torch.where(
            action_types==2,
            (advantages[..., 1] - weighted_mean_cont)/weighted_std_cont,
            0.0
        )
        advantages = torch.stack([advantages_cat[:advantages_cat.size(0)//2], advantages_cont], dim=-1)


        dataset = PPOTrainingDataset(
            observations=observations,
            action_types=action_types,
            betfracs=betfracs,
            rewards=rewards,
            masks=masks,
            action_masks=action_masks,
            advantages=advantages,
            returns=returns,
            old_log_probs=old_log_probs,
            old_log_probs_fold=old_log_probs_fold,
            legal_actions_mask=legal_actions_mask,
            advantages_fold=advantages_fold,
            weights=weight_sample,
            weights_fold=weight_fold_sample,
        )

        torch.cuda.empty_cache()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.policy.train()

        for epoch in tqdm(range(epochs)):
            epoch_policy_loss = 0.0
            epoch_entropy_loss = 0.0
            num_batches = 0

            for (
                    obs_batch,
                    actions_batch,
                    betfracs_batch,
                    masks_batch,
                    action_masks_batch,
                    legal_actions_batch,
                    adv_batch,
                    returns_batch,
                    old_log_probs_batch,
                    adv_fold_batch,
                    old_log_probs_fold_batch,
                    weights_batch,
                    weights_fold_batch,
            ) in dataloader:
                num_batches += 1

                # Combine masks for valid timesteps
                valid_mask = masks_batch & action_masks_batch

                # Forward pass through the current policy
                distribution, _ = self.policy(obs_batch, return_sequences=True, legal_actions_mask=legal_actions_batch)

                log_probs = distribution.log_prob(actions_batch, value=betfracs_batch)
                log_probs_fold = distribution.log_prob(
                    torch.zeros_like(actions_batch), value=torch.zeros_like(betfracs_batch))[...,0]

                check_tensor(log_probs)
                check_tensor(log_probs_fold)

                # Apply masks to log-probs and advantages
                log_probs = log_probs[valid_mask]
                adv_batch = adv_batch[valid_mask]
                log_probs_fold = log_probs_fold[valid_mask]
                adv_fold_batch = adv_fold_batch[valid_mask]
                # Compute PPO loss
                ratio = torch.exp(log_probs - old_log_probs_batch[valid_mask])
                ratio_fold = torch.exp(log_probs_fold - old_log_probs_fold_batch[valid_mask])
                surrogate1 = ratio * adv_batch
                surrogate2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_batch
                surrogate1_fold = ratio_fold * adv_fold_batch
                surrogate2_fold = torch.clamp(ratio_fold, 1.0 - clip_epsilon[:, 0], 1.0 + clip_epsilon[:, 0]) * adv_fold_batch

                policy_loss = -torch.min(surrogate1, surrogate2)
                policy_loss_fold = -torch.min(surrogate1_fold, surrogate2_fold)
                # two losses for action and bet-size. Betsize only relevant if action == 2
                # set betsize loss to 0 for all non-bet-actions
                policy_loss[..., 1] = torch.where(
                    actions_batch[valid_mask] != 2,
                    torch.zeros_like(policy_loss[..., 1]),
                    policy_loss[..., 1]
                )
                raise_weights = log_probs[..., 0].exp().detach()
                combined_loss = policy_loss[..., 0] + raise_weights*policy_loss[..., 1]

                # if we train with ranges, we have to weigh the samples
                action_loss = combined_loss*weights_batch[valid_mask]
                fold_loss = policy_loss_fold*weights_fold_batch[valid_mask]
                folds_mask = actions_batch[valid_mask] != 2
                action_loss = action_loss.sum() / (valid_mask.sum()-folds_mask.sum())
                fold_loss = fold_loss.sum()/(valid_mask.sum())
                reward_loss = action_loss + fold_loss
                # Compute entropy reward

                entropy = distribution.entropy()
                entropy_cat = entropy[..., 0]*entropy_coef_cat* valid_mask.float()
                entropy_size = entropy[..., 1]*entropy_coef_size* valid_mask.float()
                entropy_loss = -((entropy_cat + entropy_size)*torch.where(
                    actions_batch == 0, weights_fold_batch, weights_batch # weights_batch is 0 for folds
                )).sum()/(valid_mask.sum())

                # Total loss (policy loss + entropy reward)
                total_loss = reward_loss + entropy_loss

                # Update policy network
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                if stop_execution.is_set():
                    print("Gradient too large — stopping training!")
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=2.0)

                self.policy_optimizer.step()


        self.policy.eval()
        self.policy.to("cpu")
        self.value_function.to("cpu")



class PPOTrainingDataset(Dataset):
    def __init__(
            self,
            observations,
            action_types,
            betfracs,
            rewards,
            masks,
            action_masks,
            advantages,
            returns,
            old_log_probs,
            old_log_probs_fold,
            legal_actions_mask,
            advantages_fold,
            weights,
            weights_fold,
    ):
        """
        Dataset for PPO training data, including precomputed old log-probs.

        Args:
            observations (torch.Tensor): Shape (batch_size, max_steps, obs_dim)
            action_types (torch.Tensor): Shape (batch_size, max_steps)
            betfracs (torch.Tensor): Shape (batch_size, max_steps)
            rewards (torch.Tensor): Shape (batch_size,)
            masks (torch.Tensor): Shape (batch_size, max_steps)
            action_masks (torch.Tensor): Shape (batch_size, max_steps)
            advantages (torch.Tensor): Shape (batch_size, max_steps)
            returns (torch.Tensor): Shape (batch_size, max_steps)
            old_policy (nn.Module): The frozen old policy used to compute old log-probs.
        """
        self.observations = observations
        self.action_types = action_types
        self.betfracs = betfracs
        self.rewards = rewards
        self.masks = masks
        self.action_masks = action_masks
        self.advantages = advantages
        self.returns = returns
        self.legal_actions_mask = legal_actions_mask
        self.old_log_probs = old_log_probs
        self.old_log_probs_fold = old_log_probs_fold
        self.advantages_fold = advantages_fold
        self.weights = weights
        self.weights_fold = weights_fold


    def __len__(self):
        return self.observations.size(0)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.action_types[idx],
            self.betfracs[idx],
            self.masks[idx],
            self.action_masks[idx],
            self.legal_actions_mask[idx],
            self.advantages[idx],
            self.returns[idx],
            self.old_log_probs[idx],  # Include old log-probs in the dataset
            self.advantages_fold[idx],
            self.old_log_probs_fold[idx],
            self.weights[idx],
            self.weights_fold[idx]
        )

@torch.no_grad()
def check_hand_evals(agent, output_file="hand_evaluations.csv", new_game=None, plot_densities=False):
    """
    Sanity check for starting hands, including value function predictions,
    policy probabilities for action categories, and the most likely bet_frac.

    Args:
        value_function (nn.Module): The value function model.
        policy (nn.Module): The policy model.
        output_file (str): File path for saving results in CSV format.
    """
    if new_game is not None:
        game = new_game
    game.new_hand(first_hand=True)
    dummy_player = game.acting_player
    ranks = range(14, 1, -1)  # Ranks from Ace (14) to 2
    suits = range(4)  # Suits from 0 to 3

    observations_test = []
    hand_grouped_predictions = defaultdict(list)

    # Generate paired hands (e.g., AA, KK)
    for rank in ranks:
        for suit1, suit2 in combinations(suits, 2):
            dummy_player.holecards = [Card(rank, suit1), Card(rank, suit2)]
            state = game.get_state(dummy_player)
            dummy_state = agent.encode_state(state)
            dummy_cards = dummy_player.holecards

            observations_test.append(dummy_state.unsqueeze(dim=0))
            hand_key = f"{dummy_cards[0].representation[0]}{dummy_cards[1].representation[0]}"
            hand_grouped_predictions[hand_key].append(len(observations_test) - 1)

    # Generate non-paired hands (e.g., AKs, AKo)
    for rank1, rank2 in combinations(ranks, 2):
        for suit1 in suits:
            for suit2 in suits:
                dummy_player.holecards = [Card(rank1, suit1), Card(rank2, suit2)]
                state = game.get_state(dummy_player)
                dummy_state = agent.encode_state(state)
                dummy_cards = dummy_player.holecards
                observations_test.append(dummy_state.unsqueeze(dim=0))
                suited = "s" if suit1 == suit2 else "o"
                hand_key = f"{dummy_cards[0].representation[0]}{dummy_cards[1].representation[0]}{suited}"
                hand_grouped_predictions[hand_key].append(len(observations_test) - 1)

    # Stack observations for batch processing
    stacked_observations = torch.stack(observations_test, dim=0)

    specific_hands = ["AA", "AKo", "JTs", "75o"]
    specific_indices = []
    for hand in specific_hands:
        if hand in hand_grouped_predictions:
            specific_indices.append(hand_grouped_predictions[hand][0])  # Take the first instance
        else:
            print(f"Hand {hand} not found in the dataset.")

    # Run predictions
    value_predictions, _ = agent.value_function(stacked_observations)

    if agent.encode_value:
        stacked_observations = torch.cat([stacked_observations, value_predictions], dim=-1)
        value_predictions = value_predictions.squeeze()
    policy_distributions, _ = agent.policy(stacked_observations)

    # Extract policy distribution parameters
    category_probs = policy_distributions.category_probs
    beta_alphas = policy_distributions.beta_alphas
    beta_betas = policy_distributions.beta_betas
    beta_weights = policy_distributions.beta_weights

    # Plot Beta densities for specific hands
    if plot_densities:
        if specific_indices:
            specific_observations = stacked_observations[specific_indices]
            agent.policy.module.plot_beta_density(specific_observations, hand_labels=specific_hands)
        else:
            print("No specific hands found for plotting.")

    # Prepare CSV data
    csv_data = []
    q_values = value_predictions.shape[-1] == 2
    for hand_type, indices in hand_grouped_predictions.items():
        group_predictions = value_predictions[indices].squeeze()
        group_category_probs = category_probs[indices].squeeze()
        group_beta_alphas = beta_alphas[indices].squeeze()
        group_beta_betas = beta_betas[indices].squeeze()

        # Average prediction for the group
        if q_values:
            mean_prediction = torch.mean(group_predictions, dim=0).tolist()
        else:
            mean_prediction = [torch.mean(group_predictions, dim=0).item()]

        # Average probabilities for each action category
        mean_probs = torch.mean(group_category_probs, dim=0).tolist()

        # Expected frequency (mean of the Beta distribution)
        alpha_values = torch.mean(group_beta_alphas, dim=0)
        beta_values = torch.mean(group_beta_betas, dim=0)
        expected_frequency = torch.mean(alpha_values / (alpha_values + beta_values)).item()

        # Highest density frequency (mode or border value)
        modes = torch.zeros_like(alpha_values)
        valid_mask = (alpha_values > 1) & (beta_values > 1)
        modes[valid_mask] = (alpha_values[valid_mask] - 1) / (
                alpha_values[valid_mask] + beta_values[valid_mask] - 2
        )

        # For invalid components (alpha <= 1 or beta <= 1), assign the mode as 0 or 1 (border values)
        modes[~valid_mask & (alpha_values <= 1)] = 0  # Border value at 0
        modes[~valid_mask & (beta_values <= 1)] = 1  # Border value at 1

        # Average the modes for the group
        highest_density_frequency = torch.mean(modes).item()

        # Add row to CSV data
        csv_data.append([
            hand_type, *mean_prediction, *mean_probs,
            expected_frequency, highest_density_frequency
        ])

    # Sort by value prediction (descending)
    csv_data.sort(key=lambda x: x[1], reverse=True)
    columns = ["Hand Type"]
    if agent.encode_value:
        columns += ["Q Check/Call", "Q Bet/Raise"]
    else:
        columns.append("Mean Value")
    columns += ["Fold Prob", "Check/Call Prob","Bet/Raise Prob"]

    columns += [ "Expected Frequency", "Highest Density Frequency"]

    # Write to CSV
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(csv_data)



if __name__ == "__main__":
    bb = 10
    exploration_temp = 1.2
    load_models = False
    save_path = "../policies/saved_models"
    save_path_value = "../value_functions/saved_models"
    episodes_per_epoch = 10000
    stacks = [bb*2, bb*2]

    # Initialize the game
    game = Game(n_players=2, stacks=stacks, hand_history=True)
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

    agents = {player:
        Agent(
            player,
            MixedGruPolicy(
                input_size=state_vector_value + value_input,
                hidden_size=256,
                num_gru_layers=1,
                linear_layers=(256, 128),
            ),
            # Initialize policies and value function with the determined state vector size
            GRUValueFunction(
                input_size=state_vector_value,
                hidden_size=256,
                num_gru_layers=1,
                linear_layers=(256, 128),
                output_dim=output,
            ),
            bool(value_input)
        )
        for player, output, value_input in zip(game.players, (2,2), (2, 2))
    }
    if load_models:
        for agent_nr, agent in enumerate(agents.values()):

            policy_state_dict = torch.load(f"../policies/saved_models/policy_{agent_nr}.pt", map_location="cpu", weights_only=True)
            with torch.no_grad():
                agent.policy.load_state_dict(policy_state_dict)
                agent.policy.to("cpu")  # Ensure policy is explicitly on CPU
                #agent.policy = torch.compile(agent.policy)
                agent.policy.module.gru.gru.flatten_parameters()
                agent.policy.eval()
            # Load optimizer state and transfer to the correct device
            optimizer_state_dict = torch.load(f"../policies/saved_models/optimizer_{agent_nr}.pt", map_location=device, weights_only=True)
            agent.policy_optimizer.load_state_dict(optimizer_state_dict)

            # Ensure optimizer state tensors are on the correct device
            for state in agent.policy_optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)


            agent.value_function.load_state_dict(
                torch.load(f"../value_functions/saved_models/model_{agent_nr}.pt", weights_only=True))
            # value_function = torch.compile(value_function)
            agent.value_optimizer.load_state_dict(
                torch.load(f"../value_functions/saved_models/optimizer_{agent_nr}.pt", weights_only=True))
            for state in agent.value_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)


    e = 0
    while True:
        # Progress bar for episodes in the current epoch
        for a_nr, agent in enumerate(agents.values()):
            agent.clear_episode_buffer()
            gc.collect()
            agent.value_function.to("cpu")

            check_hand_evals(agent, output_file=f"evaluations/hand_evaluations_{a_nr}_{e:04}.csv", new_game=game)

        # Initialize the counter and timing for env_steps
        env_steps = 0
        start_time = time.time()

        with tqdm(total=episodes_per_epoch, desc="Episodes") as pbar:
            for _ in range(episodes_per_epoch):
                # Start a new hand and create an Episode object for each player
                game.new_hand(first_hand=False)
                for agent in agents.values():
                    # check blinds
                    agent.blind = agent.player.bet
                    agent.episodes.append(Episode())
                    agent.policy.reset()

                while not game.finished:
                    # Record observations for all players at this step
                    for agent in agents.values():
                        game_state = encode_state(game, agent.player)
                        hand_strength = encode_strength(agent.player.holecards, game.board)
                        state = np.hstack([game_state, hand_strength])
                        agent.episodes[-1].add_observation(state)
                        if agent.player != game.acting_player:
                            agent.episodes[-1].add_action(None, None)
                        agent.add_to_sequence(state)

                    # Acting player decides an action
                    current_player = game.acting_player
                    action, betsize, action_type, betfrac, legal_actions = agents[current_player].get_action(game=game)
                    action_add = (action_type, betfrac)
                    env_steps += 1  # Count the env step
                    agents[current_player].episodes[-1].add_action(action_add, legal_actions)

                    game.implement_action(current_player, action, betsize)


                # At the end of the hand, record the rewards for all players
                # If
                for player in game.players:
                    reward = player.stack - player.starting_stack + agent.blind
                    agents[player].episodes[-1].finish_episode(reward)

                # Update the progress bar
                elapsed_time = time.time() - start_time
                env_steps_per_second = env_steps / elapsed_time
                pbar.update(1)
                pbar.set_postfix({"env_steps/s": f"{env_steps_per_second:.2f}"})

        # Create training data for all players
        for agent in agents.values():
            agent.generate_training_data()


        for nr, agent in enumerate(agents.values()):
            agent.train_policy(epochs=5, batch_size=64, entropy_coef=0.1, clip_epsilon=0.1)
            torch.save(agent.policy.state_dict(), f"{save_path}/policy_{nr}.pt")
            torch.save(agent.policy_optimizer.state_dict(), f"{save_path}/optimizer_{nr}.pt")
            agent.policy.to("cpu")
            if agent.value_function.output_dim ==2:
                with torch.no_grad():
                    state_input = agent.training_data["observations"]
                    value_predictions = agent.value_function(state_input, return_sequences=True)[0]
                    state_input = torch.cat([state_input, value_predictions], dim=-1)
                    probs = agent.policy(state_input, return_sequences=True)[0].category_probs
                action_types = agent.training_data["action_types"]
                action_masks = agent.training_data["action_masks"]
            else:
                probs = None
                action_types = None
                action_masks = None
            agent.value_function.train_on_data(
                optimizer=agent.value_optimizer,
                epochs=5,
                observations=agent.training_data["observations"],
                rewards=agent.training_data["rewards"],
                mask=agent.training_data["masks"],
                actions=action_types,
                action_masks=action_masks,
                probs=probs,
            )
            torch.save(agent.value_function.state_dict(), f"{save_path_value}/model_{nr}.pt")
            torch.save(agent.value_optimizer.state_dict(), f"{save_path_value}/optimizer_{nr}.pt")

        torch.cuda.empty_cache()
        e+=1
