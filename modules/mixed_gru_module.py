from modules.gru_module import GRUModule
from torch import nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Beta, Categorical

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Beta, Categorical
class UnifiedDistribution(torch.distributions.Distribution):
    def __init__(
            self,
            category_logits,
            beta_alphas,
            beta_betas,
            beta_logits,
            categorical_mask=None,
    ):
        """
        A unified distribution for both discrete actions and continuous betting, using Beta distributions.

        Args:
            category_logits (Tensor): Logits for each category (shape: (B, 5)).
            beta_alphas (Tensor): Alpha parameters for Beta components (shape: (B, num_betas)).
            beta_betas (Tensor): Beta parameters for Beta components (shape: (B, num_betas)).
            beta_logits (Tensor): Logits for mixture weights (shape: (B, num_betas)).
        """
        super().__init__()
        if categorical_mask is None:
            categorical_mask = torch.ones_like(category_logits, dtype=torch.bool)

        # Step 2: Mask logits for invalid categories (set to -inf for invalid actions)
        mask_value = torch.tensor(
            torch.finfo(category_logits.dtype).min, dtype=category_logits.dtype
        )
        self.category_logits = torch.where(categorical_mask, category_logits, mask_value)

        self.beta_alphas = beta_alphas
        self.beta_betas = beta_betas
        self.beta_logits = beta_logits

        # Create Beta mixture components
        self.beta_components = Beta(self.beta_alphas, self.beta_betas)
        self.beta_mixture = Categorical(logits=self.beta_logits)
        self.raise_category = self.category_logits.shape[-1] - 1

    def get_category_probs(self, temperature=1.0):
        """
        Compute category probabilities from logits with optional temperature scaling.

        Args:
            temperature (float): Temperature for scaling logits.

        Returns:
            Tensor: Probabilities for each category (shape: (B, 5)).
        """
        if hasattr(self, 'mask_value'):  # Ensure mask_value is defined
            scaled_logits = torch.where(
                self.category_logits == self.mask_value,  # Preserve masked logits
                self.category_logits,  # Keep masked logits unchanged
                self.category_logits / temperature  # Scale valid logits
            )
        else:
            scaled_logits = self.category_logits / temperature
        return F.softmax(scaled_logits, dim=-1)

    @property
    def category_probs(self):
        """
        Lazily compute category probabilities from logits.

        Returns:
            Tensor: Probabilities for each category (shape: (B, 5)).
        """
        return F.softmax(self.category_logits, dim=-1)

    @property
    def beta_weights(self):
        return F.softmax(self.beta_logits, dim=-1)

    def sample(self, temperature=1.0):
        """
        Sample from the unified distribution.

        Returns:
            Tensor: Sampled categories (shape: (B, T)).
            Tensor: Sampled actions (shape: (B, T)).
        """
        # Get batch size and number of timesteps

        batch_size, num_timesteps, num_categories = self.category_logits.shape  # (B, T, num_categories)
        if hasattr(self, 'mask_value'):  # Ensure mask_value is defined
            scaled_logits = torch.where(
                self.category_logits == self.mask_value,  # Preserve masked logits
                self.category_logits,  # Keep masked logits unchanged
                self.category_logits / temperature  # Scale valid logits
            )
        else:
            scaled_logits = self.category_logits / temperature
            # Sample categories using logits
        category = Categorical(logits=scaled_logits).sample()  # (B, T)

        # Generate Beta samples for category 4 (bet/raise)
        beta_samples = self.beta_components.sample()  # (B, T, num_betas)
        beta_samples = torch.clamp(beta_samples, min=1e-8, max=1.0 - 1e-8)  # Avoid boundaries

        # Select a mixture component
        beta_idx = self.beta_mixture.sample()  # (B, T)
        selected_beta_samples = beta_samples[
            torch.arange(batch_size).unsqueeze(-1),  # Broadcast batch indices
            torch.arange(num_timesteps),  # Broadcast timestep indices
            beta_idx
        ]  # (B, T)

        # Assign actions based on categories
        fractions = torch.zeros_like(category, dtype=torch.float32, device=self.category_logits.device)  # Initialize
        fractions[category == self.raise_category] = selected_beta_samples[category == self.raise_category]  # Continuous bet/raise
        fractions[category == 0] = 0  # Fold
        fractions[category == 1] = 0  # Check/call
        if self.raise_category == 4:
            fractions[category == 2] = 0  # Minbet
            fractions[category == 3] = 1  # All-in

        return category, fractions

    def log_prob(self, category, value=None, temperature=1.0):
        """
        Compute log-probabilities for a given category and optional continuous value.

        Args:
            category (Tensor): Discrete category (shape: (B, T)).
            value (Tensor, optional): Continuous values (shape: (B, T)) for category 4.
            temperature:

        Returns:
            Tensor: Log probabilities (shape: (B, T)).

        """
        eps = 1e-8  # Small constant to prevent numerical issues

        # Compute categorical log probabilities using logits
        if hasattr(self, 'mask_value'):  # Ensure mask_value is defined
            scaled_logits = torch.where(
                self.category_logits == self.mask_value,  # Preserve masked logits
                self.category_logits,  # Keep masked logits unchanged
                self.category_logits / temperature  # Scale valid logits
            )
        else:
            scaled_logits = self.category_logits / temperature
            # Sample categories using logits
        categorical_log_prob = Categorical(logits=scaled_logits).log_prob(category)  # (B, T)
        min_log_prob = torch.log(torch.tensor(eps, device=self.category_logits.device))
        categorical_log_prob = torch.clamp(categorical_log_prob, min=min_log_prob)

        beta_log_probs = torch.zeros_like(self.beta_logits)
        value = torch.clamp(value, min=eps, max=1.0 - eps)  # Clamp value to avoid boundary issues

        # Compute Beta log-probs
        beta_log_probs[category==self.raise_category] = self.beta_components.log_prob(value.unsqueeze(-1))[category==self.raise_category]  # (B, T, num_betas)

        # Combine Beta log-probabilities with mixture weights (from logits)
        beta_weights = F.softmax(self.beta_logits, dim=-1)  # Convert logits to probabilities
        weighted_beta_log_probs = beta_log_probs + torch.log(beta_weights + eps)  # (B, T, num_betas)

        # Compute final Beta log-prob with logsumexp
        beta_log_prob = torch.logsumexp(weighted_beta_log_probs, dim=-1)  # (B, T)
        return torch.stack([categorical_log_prob, beta_log_prob], dim=-1)

    def entropy(self):
        """
        Compute the entropy of the unified distribution.

        Returns:
            Tensor: Entropy values (shape: (B, T)).
        """
        # Categorical entropy using logits
        categorical_entropy = Categorical(logits=self.category_logits).entropy()  # Shape: (B, T)

        # Beta component entropies
        beta_entropies = self.beta_components.entropy()  # Shape: (B, T, num_betas)
        beta_weights = F.softmax(self.beta_logits, dim=-1)  # Convert logits to probabilities
        weighted_beta_entropy = torch.sum(
            beta_weights * beta_entropies, dim=-1
        )  # Weighted sum over num_betas, Shape: (B, T)

        # Entropy of the Beta mixture weights (from logits)
        beta_mixture_entropy = -torch.sum(
            beta_weights * torch.log(beta_weights + 1e-8), dim=-1
        )  # Reduce over num_betas, Shape: (B, T)

        # Total Beta entropy: component entropy + mixture entropy
        beta_entropy = weighted_beta_entropy + beta_mixture_entropy  # Shape: (B, T)

        # Combine categorical and Beta entropies for category 4
        combined_entropy = categorical_entropy + beta_entropy*self.category_probs[:, :, -1]
        return combined_entropy


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
        self.category_layer = nn.Linear(current_input_size, categories)  # 5 categories: fold, check/call, minbet, allin, raise

        # Initialize Beta parameters layer
        self.beta_layer = nn.Linear(current_input_size, 2 * num_peaks)  # Output alphas and betas

        # Custom initialization: Start around 1 (uniform distribution)
        torch.nn.init.constant_(self.beta_layer.bias[:num_peaks], 1.0)  # Initialize alphas
        torch.nn.init.constant_(self.beta_layer.bias[num_peaks:], 1.0)  # Initialize betas
        # Outputs alpha and beta for each peak
        self.beta_weight_layer = nn.Linear(current_input_size, num_peaks)  # Mixture weights for the Beta components

        self.activation = activation()
        self.num_peaks = num_peaks

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
        category_logits = self.category_layer(out)  # (B, T, num_categories)

        # Compute Beta parameters (alphas and betas) using softplus
        beta_params = nn.functional.softplus(self.beta_layer(out))  # (B, T, num_betas * 2)
        alphas, betas = torch.chunk(beta_params, 2, dim=-1)  # Ensure positivity with softplus

        # Compute mixture logits for Beta mixture components
        beta_logits = self.beta_weight_layer(out)  # (B, T, num_betas)

        # Create unified distribution
        unified_distribution = UnifiedDistribution(
            category_logits=category_logits,  # Pass logits directly
            beta_alphas=alphas,
            beta_betas=betas,
            beta_logits=beta_logits,  # Pass logits directly
            categorical_mask=action_mask
        )

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

