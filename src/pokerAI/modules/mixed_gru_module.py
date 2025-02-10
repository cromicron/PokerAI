from modules.gru_module import GRUModule
from torch import nn
import matplotlib.pyplot as plt
from modules.mixed_distribution_head import MixedDistributionHead
import torch



class MixedGRUModule(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_gru_layers,
            linear_layers,
            num_peaks=3,  # Number of Beta components
            activation=nn.LeakyReLU,
            single_raise=True,
    ):
        super().__init__()

        # GRU Layer
        self.gru = GRUModule(input_size, hidden_size, num_gru_layers, linear_layers, activation, output_dim=None)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                # Apply Xavier (Glorot) Uniform Initialization to all weight matrices
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize biases to zero (optional, can be constant or other strategies too)
                torch.nn.init.zeros_(param)
        current_input_size = linear_layers[-1]
        categories = 3 if single_raise else 5

        # Output Layers
        self.head = MixedDistributionHead(current_input_size, 2 * num_peaks, activation)
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
        out, hidden_state = self.gru(x, hidden_state,
                                     return_sequences)  # (B, T, hidden_size), (num_layers, B, hidden_size)

        # Compute category logits
        unified_distribution = self.head(out)

        return unified_distribution, hidden_state

    def plot_beta_density(self, input_tensor, hand_labels=None, num_points=1000):
        """
        Generate density plots of the Beta mixture distribution for a batch of inputs,
        with hand representations included in the labels.

        Args:
            input_tensor (Tensor): Input tensor to the model (B, input_size).
            hand_labels (list of str): Hand representations for each observation (e.g., ["AA", "AKo"]).
            num_points (int): Number of points to calculate density over the range [0, 1].
        """
        # Forward pass to get the unified distribution
        unified_distribution, _ = self(input_tensor)

        # Extract Beta parameters for the batch and squeeze away all singleton dimensions
        beta_alphas = unified_distribution.beta_alphas.squeeze()  # Remove all singleton dims
        beta_betas = unified_distribution.beta_betas.squeeze()  # Remove all singleton dims
        beta_weights = unified_distribution.beta_weights.squeeze()  # Remove all singleton dims

        # Ensure the remaining shapes are valid
        if beta_alphas.dim() != 2 or beta_betas.dim() != 2 or beta_weights.dim() != 2:
            raise ValueError(
                "Expected Beta parameters to have shape (B, num_components) after squeezing."
            )

        # Generate x values for density calculation
        x = torch.linspace(0, 1, num_points, dtype=torch.float32)  # Shape: (num_points,)

        plt.figure(figsize=(10, 6))

        # Iterate over batch
        for i in range(beta_alphas.size(0)):
            alphas = beta_alphas[i]  # Shape: (num_components,)
            betas = beta_betas[i]  # Shape: (num_components,)
            weights = beta_weights[i]  # Shape: (num_components,)

            # Initialize mixture density for the current observation
            mixture_density = torch.zeros_like(x)  # Shape: (num_points,)

            # Compute the density for the Beta mixture model
            for j in range(alphas.size(0)):  # Iterate over mixture components
                alpha = alphas[j].item()  # Extract scalar value for alpha
                beta = betas[j].item()  # Extract scalar value for beta
                weight = weights[j].item()  # Extract scalar value for weight

                # Avoid invalid parameters (alpha, beta > 0)
                if alpha <= 0 or beta <= 0:
                    continue

                # Calculate the Beta PDF for each component using torch.lgamma
                log_beta = (
                        torch.lgamma(torch.tensor(alpha))
                        + torch.lgamma(torch.tensor(beta))
                        - torch.lgamma(torch.tensor(alpha + beta))
                )
                beta_pdf = (
                        (x ** (alpha - 1)) * ((1 - x) ** (beta - 1)) / torch.exp(log_beta)
                )

                # Add weighted contribution to the mixture
                mixture_density += weight * beta_pdf

            # Determine label for the observation
            label = hand_labels[i] if hand_labels and i < len(hand_labels) else f"Observation {i + 1}"

            # Plot the mixture density for this observation
            plt.plot(
                x.numpy(),
                mixture_density.numpy(),
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

