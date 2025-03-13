from pokerAI.modules.split_gru_module import SplitGRUModule
from pokerAI.modules.mixed_distribution_head import MixedDistributionHead
from torch import nn
import matplotlib.pyplot as plt
import torch


class SplitGRUActionModule(nn.Module):
    def __init__(
            self,
            input_size_recurrent,
            input_size_regular,
            hidden_size,
            num_gru_layers,
            linear_layers,
            linear_layers_cat=(128,),
            linear_layers_cont=(128,),
            num_peaks=3,
            activation=nn.GELU,
    ):
        super().__init__()
        self._input_size_recurrent = input_size_recurrent
        self._input_size_regular = input_size_regular
        # GRU Layer for processing betting sequences only
        self.split_gru = SplitGRUModule(
            input_size_recurrent,
            input_size_regular,
            hidden_size,
            num_gru_layers,
            linear_layers,
            activation,
        )

        self.head = MixedDistributionHead(
            input_size=linear_layers[-1],
            num_components=num_peaks,
            layers_cat=linear_layers_cat,
            layers_cont=linear_layers_cont,
        )
        self.activation = activation()



    def forward(self, x, hidden_state=None, return_sequences=False, action_mask=None):
        """
        Forward pass for MixedGRUModule with Beta distributions.

        Args:
            x: Input tensor (B, T, input_size).
            hidden_state: Hidden state for the GRU (num_layers, B, hidden_size).
            return_sequences (bool): If True, return outputs for all timesteps. If False, return only the last timestep.

        Returns:
            UnifiedDistribution: A unified distribution object for sampling and log-probability.
            Tensor: Updated hidden state.
        """
        # GRU forward
        out, hidden_state = self.split_gru(x, hidden_state, return_sequences)  # GRU handles all layers internally
        dist = self.head(out, action_mask)
        return dist, hidden_state

    def plot_beta_density(self, input_tensor, hand_labels=None, num_points=100000):
        """
        Generate density plots of the Beta mixture distribution for a batch of inputs,
        with hand representations included in the labels.

        Args:
            input_tensor (Tensor): Input tensor to the model (B, input_size).
            hand_labels (list of str): Hand representations for each observation (e.g., ["AA", "AKo"]).
            num_points (int): Number of points to calculate density over the range [0, 1].
        """
        # Forward pass to get the unified distribution
        dist, _ = self(input_tensor.repeat_interleave(num_points, dim=0) , return_sequences=False)

        # Generate x values for density calculation
        x = torch.linspace(1e-4, 1-1e-4, num_points, dtype=torch.float32).repeat(input_tensor.size(0)).unsqueeze(1)
        mixture_density = torch.exp(dist.log_prob(torch.full_like(x, 2), x)[..., 1])
        x =   x ** (16.6**0.5)
        plt.figure(figsize=(10, 6))

        # Iterate over batch
        for i in range(input_tensor.size(0)):


            # Determine label for the observation
            label = hand_labels[i] if hand_labels and i < len(hand_labels) else f"Observation {i + 1}"

            # Plot the mixture density for this observation
            plt.plot(
                x[i*num_points:(i+1)*num_points].numpy(),
                mixture_density[i*num_points:(i+1)*num_points].numpy(),
                label=label,
                alpha=0.7,
            )


        # Add plot details
        plt.title("Beta Mixture Density Plot", fontsize=14)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid()
        plt.show()


    def plot_gaussian_density(self, input_tensor, hand_labels=None, num_points=100000):
        """
        Generate density plots of the Gaussian mixture distribution for a batch of inputs,
        with hand representations included in the labels.

        Args:
            input_tensor (Tensor): Input tensor to the model (B, input_size).
            hand_labels (list of str): Hand representations for each observation (e.g., ["AA", "AKo"]).
            num_points (int): Number of points to calculate density over the range [0, 1].
        """
        # Forward pass to get the unified distribution
        unified_distribution, _ = self(input_tensor)

        # Generate x values for density calculation (stay within (0,1))
        x = torch.linspace(-1, 2, num_points, dtype=torch.float32).to(input_tensor.device)
        log_probs = unified_distribution.gaussian_dist.log_prob(x.unsqueeze(0))
        pdf_vals = torch.exp(log_probs).squeeze(0).cpu().numpy()
        plt.figure(figsize=(10, 6))

        # Iterate over batch
        for i in range(input_tensor.size(0)):
            label = hand_labels[i] if hand_labels and i < len(hand_labels) else f"Observation {i + 1}"
            plt.plot(x.cpu().numpy(), pdf_vals[i], label=label, alpha=0.7)

        # Add plot details
        plt.title("Gaussian Mixture Density Plot", fontsize=14)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid()
        plt.show()

