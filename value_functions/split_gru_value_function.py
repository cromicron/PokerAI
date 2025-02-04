from value_functions.gru_value_function import PokerDataset
from modules.split_gru_module import SplitGRUModule
import torch
import torch.nn as nn

class GRUValueFunction(SplitGRUModule):
    def __init__(
            self,
            input_size_recurrent,
            input_size_regular,
            hidden_size,
            num_gru_layers,
            linear_layers,
            activation=nn.GELU,
            output_dim=2,
            idx_stack_start=61,
            idx_stack_now=30,
            idx_bet=41,
    ):
        linear_layers = [layer for layer in linear_layers]
        super().__init__(
            input_size_recurrent,
            input_size_regular,
            hidden_size,
            num_gru_layers,
            linear_layers,
            activation,
        )
        self.output_dim = output_dim
        self.output_layer = nn.Linear(linear_layers[-1], output_dim)
        self.idx_stack_start=idx_stack_start
        self.idx_stack_now=idx_stack_now
        self.idx_bet=idx_bet

    def forward(self, x, hidden_state=None, return_sequences=False):
        out, hidden = super().forward(x, hidden_state, return_sequences)
        return self.output_layer(out), hidden



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