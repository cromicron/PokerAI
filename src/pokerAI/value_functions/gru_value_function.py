import torch
import torch.nn as nn
from pokerAI.modules.gru_module import GRUModule
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PokerDataset(Dataset):
    def __init__(
            self,
            observations,
            rewards,
            mask,
            value_function,
            action_categories=None,
            action_masks=None,
            probs=None,
            weights=None,
            predictions=None
    ):
        """
        Dataset for poker training data. Automatically computes next-step value targets.

        Args:
            observations (torch.Tensor): Observations of shape (batch_size, max_steps, obs_dim).
            rewards (torch.Tensor): Rewards at the end of each episode (batch_size, max_steps).
            mask (torch.Tensor): Mask indicating valid timesteps (batch_size, max_steps).
            value_function (nn.Module): The value function model used to compute next-step predictions.
        """
        self.observations = observations.to(device)
        self.mask = mask.to(device)
        if weights is not None:
            self.weights = weights.to(device)
        # Compute targets
  # (batch_size, max_steps, 1)
 # Remove last dim, shape now: (batch_size, max_steps)
        assert value_function.output_dim in (1,2), "value function output must be 1 for value function and 2 for q"
        if value_function.output_dim == 1:
            self.targets = torch.roll(predictions, shifts=-1, dims=1)  # Shift targets for next-step prediction
            for i in range(rewards.size(0)):
                # Identify the last valid timestep using the sequence mask
                last_valid_idx = mask[i].nonzero()[-1].item()
                self.targets[i, last_valid_idx] = rewards[i]
        elif value_function.output_dim ==2:
            assert action_categories is not None, "you must provide actions for q-function"
            assert probs is not None, "you must provide probabilities for actions for q-function"
            assert action_masks is not None, "you must provide action masks for actions for q-function"
            action_categories = action_categories.to(device)
            action_masks = action_masks.to(device)
            probs = probs.to(device)
            rewards = rewards.to(device)
            # values are the average of the next state, when an action is taken
            # true values for folds can be caluclated by stack-starting_stack + blind
            blind = observations[:, 0, value_function.idx_bet]
            starting_stack = observations[..., value_function.idx_stack_start]
            current_stack = observations[..., value_function.idx_stack_now]
            q_fold = (current_stack - starting_stack + blind.unsqueeze(-1)).unsqueeze(-1)
            predictions = torch.cat([q_fold.to(device), predictions], dim=-1)
            values = (predictions * probs).sum(dim=-1)
            n, m = values.size()
            last_actions = torch.where(action_masks, torch.arange(m, device=device), -1).max(dim=1).values
            # Step 1: Create indices for columns
            indices = torch.arange(m, device=device).expand(n, m)  # Row-wise indices
            # Step 2: Replace invalid positions with a large index (out-of-bounds)
            masked_indices = torch.where(action_masks, indices, m)  # Non-True positions set to "out-of-bounds"
            # Step 3: Compute the next valid index for each position
            # Create a large tensor for forward-filling the valid indices
            next_indices = torch.full((n, m), m, dtype=torch.long, device=device)  # Initialize all with "out-of-bounds"
            for i in range(m - 2, -1, -1):  # Traverse backward
                next_indices[:, i] = torch.where(action_masks[:, i + 1], masked_indices[:, i + 1],
                                                 next_indices[:, i + 1])
            # Step 4: Replace values with the next valid value
            next_values = torch.gather(values, 1, next_indices.clamp(max=m - 1))
            result = torch.where(action_masks, next_values, values)
            result[torch.arange(n, device=device), last_actions] = rewards


            # replace q values with the appropriate values
            rows, cols = torch.where(action_masks)
            predictions[rows, cols, action_categories[rows, cols]] = result[rows, cols]


            self.targets = predictions[..., 1:]
            self.mask = action_masks


    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        if self.weights is not None:
            item = (self.observations[idx], self.targets[idx], self.mask[idx], self.weights[idx])
        else:
            item = (self.observations[idx], self.targets[idx], self.mask[idx])
        return item


class GRUValueFunction(GRUModule):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_gru_layers,
            linear_layers,
            activation=nn.LeakyReLU,
            output_dim=2,
            output_activation=None,
            idx_stack_start=103,
            idx_stack_now=94,
            idx_bet=95,
    ):
        self.output_dim = output_dim
        super(GRUValueFunction, self).__init__(
            input_size,
            hidden_size,
            num_gru_layers,
            linear_layers,
            activation=activation,
            output_dim=output_dim,
            output_activation=output_activation,
        )
        self.idx_stack_start=idx_stack_start
        self.idx_stack_now=idx_stack_now
        self.idx_bet=idx_bet

    def loss(self, batch):
        pass

    def train_on_data(
            self,
            optimizer,
            epochs: int,
            observations: torch.Tensor,
            rewards: torch.Tensor,
            mask: torch.Tensor,
            batch_size: int = 32,
            actions=None,
            probs=None,
            action_masks=None,
            verbose=True,
    ):
        """
        Trains the value function using predicted values of the next step as targets.

        Args:
            epochs (int): Number of training epochs.
            observations (torch.Tensor): Padded observations of shape (batch_size, max_steps, obs_dim).
            rewards (torch.Tensor): Rewards at the end of each episode (batch_size, max_steps).
            mask (torch.Tensor): Mask indicating valid timesteps (batch_size, max_steps).
            batch_size (int): Batch size for training.
        """
        self.to(device)
        self.train()
        dataset = PokerDataset(
            observations=observations,
            rewards=rewards,
            mask=mask,
            value_function=self,
            action_categories=actions,
            action_masks=action_masks,
            probs=probs,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_fn = torch.nn.MSELoss(reduction='none')  # Element-wise loss

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                optimizer.zero_grad()

                batch_observations, batch_targets, batch_mask = batch

                # Forward pass for predictions
                predictions, _ = self(batch_observations, return_sequences=True) # (batch_size, max_steps)


                # Compute loss with masking
                raw_loss = loss_fn(predictions.squeeze(), batch_targets)
                if raw_loss.size(-1) == 2:
                    raw_loss = raw_loss.max(dim=-1).values# (batch_size, max_steps)
                masked_loss = raw_loss * batch_mask.float()  # Apply mask to ignore invalid timesteps
                loss = masked_loss.sum() / batch_mask.sum()

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {epoch_loss / num_batches:.4f}")
        torch.cuda.empty_cache()
        self.to("cpu")
        self.eval()




