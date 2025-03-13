import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution, Beta, Categorical

import threading

# Create a threading event (this is shared across threads)
stop_execution = threading.Event()
def check_tensor(name):
    def hook(grad):
        global stop_execution
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print(f"NAN Gradient issue in {name}")
        if abs(grad).any() > 0:
            stop_execution.append(grad)
            print(f"Large Gradient issue in {name}")
        #print(f"max grad {torch.max(grad)}")
    return hook

class UnifiedDistribution(Distribution):
    def __init__(
            self,
            category_logits,
            beta_alphas,
            beta_betas,
            beta_logits,
            categorical_mask=None,
            min_fraction=1e-6,
            max_fraction=1-1e-6,
    ):
        """
        A unified distribution for both discrete actions and continuous betting, using Beta distributions.

        Args:
            category_logits (Tensor): Logits for each category (shape: (B, 5)).
            beta_alphas (Tensor): Alpha parameters for Beta components (shape: (B, num_betas)).
            beta_betas (Tensor): Beta parameters for Beta components (shape: (B, num_betas)).
            beta_logits (Tensor): Logits for mixture weights (shape: (B, num_betas)).
            compression_factor (Tensor): Regulates how much effort it takes to move to higher values
        """
        super().__init__()
        if categorical_mask is None:
            categorical_mask = torch.ones_like(category_logits, dtype=torch.bool)

        # Step 2: Mask logits for invalid categories (set to -inf for invalid actions)
        self.mask_value = torch.tensor(
            torch.finfo(category_logits.dtype).min, dtype=category_logits.dtype
        )
        self.category_logits = torch.where(categorical_mask, category_logits, self.mask_value)
        self.beta_alphas = beta_alphas
        self.beta_betas = beta_betas
        min_beta = 0.001  # Prevent deterministic collapse


        self.beta_alphas.data.clamp_(min=min_beta)
        self.beta_betas.data.clamp_(min=min_beta)

        self.beta_logits = beta_logits

        # Create Beta mixture components
        self.beta_components = Beta(self.beta_alphas, self.beta_betas)
        self.beta_mixture = Categorical(logits=self.beta_logits)
        self.raise_category = self.category_logits.shape[-1] - 1
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction


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

        Args:
            temperature (float): Softmax/Beta temperature (>0).
                                 - 1.0 = no change
                                 - >1.0 = “flatten” or less peaked
                                 - <1.0 = “sharpen” or more peaked

        Returns:
            category (Tensor): Sampled categories (shape: (B, T)).
            fractions (Tensor): Sampled continuous bet/raise fraction (shape: (B, T)).
        """
        # ------------------------------------------------
        # 1. SAMPLE THE CATEGORICAL ACTION
        # ------------------------------------------------
        batch_size, num_timesteps, num_categories = self.category_logits.shape  # (B, T, num_categories)

        # If there's a mask_value, do not temperature-scale masked logits; only the valid ones
        if hasattr(self, 'mask_value'):
            scaled_logits = torch.where(
                self.category_logits == self.mask_value,
                self.category_logits,  # Keep masked logits as is
                self.category_logits / temperature  # Scale valid logits
            )
        else:
            scaled_logits = self.category_logits / temperature

        # Sample the discrete category
        category_dist = Categorical(logits=scaled_logits)
        category = category_dist.sample()  # (B, T)

        # ------------------------------------------------
        # 2. SAMPLE FROM THE (MIXTURE OF) BETA DISTRIBUTIONS
        # ------------------------------------------------
        # "beta_components" presumably is a PyTorch Beta distribution object
        #    with shape (B, T, num_betas) or is batched accordingly.

        if temperature == 1.0:
            # No temperature change => sample directly
            beta_samples = self.beta_components.sample()  # shape (B, T, num_betas)
        else:
            # Re-temper the Beta distribution parameters
            # If self.beta_components is a Beta distribution,
            #   it has .concentration0 (alpha) and .concentration1 (beta) internally
            alpha_orig = self.beta_components.concentration0  # shape (B, T, num_betas)
            beta_orig = self.beta_components.concentration1  # shape (B, T, num_betas)

            # Using alpha_temp = 1 + (alpha_orig - 1)/temperature
            alpha_temp = 1.0 + (alpha_orig - 1.0) / temperature
            beta_temp = 1.0 + (beta_orig - 1.0) / temperature

            # Create a new Beta distribution with tempered parameters
            tempered_beta_components = Beta(alpha_temp, beta_temp)
            beta_samples = tempered_beta_components.sample()  # shape (B, T, num_betas)

        # Clamp away from exact 0 or 1 to avoid potential numerical issues
        beta_samples = torch.clamp(beta_samples, min=self.min_fraction, max=self.max_fraction)

        # ------------------------------------------------
        # 3. SELECT WHICH BETA COMPONENT IS USED
        # ------------------------------------------------
        beta_idx = self.beta_mixture.sample()  # (B, T) which mixture index to pick
        # Gather the selected Beta samples across the last dimension (num_betas)
        selected_beta_samples = beta_samples[
            torch.arange(batch_size).unsqueeze(-1),
            torch.arange(num_timesteps),
            beta_idx
        ]  # shape (B, T)

        # ------------------------------------------------
        # 4. ASSIGN FRACTIONS (BET AMOUNTS) BASED ON CATEGORY
        # ------------------------------------------------
        fractions = torch.zeros_like(category, dtype=torch.float32, device=self.category_logits.device)

        # If the "raise_category" means a bet/raise is chosen, use continuous fraction
        fractions[category == self.raise_category] = selected_beta_samples[category == self.raise_category]

        # Otherwise, set fractions to 0 for fold, call, etc.
        fractions[category == 0] = 0  # Fold
        fractions[category == 1] = 0  # Check/Call
        if self.raise_category == 4:
            # Example handling for minbet/all-in as separate discrete categories
            fractions[category == 2] = 0  # Minbet
            fractions[category == 3] = 1  # All-in

        return category, fractions


    def log_prob(self, category, value=None, temperature=1.0):
        """
        Compute log-probabilities for a given category and optional continuous value.
        """

        eps = 1e-8  # Small constant to prevent numerical issues

        # Compute categorical log probabilities using logits
        if hasattr(self, 'mask_value'):
            scaled_logits = torch.where(
                self.category_logits == self.mask_value,
                self.category_logits,
                self.category_logits / temperature
            )
        else:
            scaled_logits = self.category_logits / temperature

        categorical_log_prob = Categorical(logits=scaled_logits).log_prob(category)
        categorical_log_prob = torch.clamp(categorical_log_prob, min=torch.log(torch.tensor(eps, device=self.category_logits.device)))
        categorical_log_prob = torch.where(
            (torch.gather(scaled_logits, 2, value.unsqueeze(-1).long()) == self.mask_value).squeeze(-1),
            0,
            categorical_log_prob
        )
        beta_log_probs = torch.zeros_like(self.beta_logits)
        raises = category == self.raise_category


        beta_log_probs = torch.where(
            raises.unsqueeze(-1),
            self.beta_components.log_prob(value.unsqueeze(-1)),
            beta_log_probs
        )
        # Combine Beta log-probabilities with mixture weights
        beta_weights = F.softmax(self.beta_logits, dim=-1)

        weighted_beta_log_probs = beta_log_probs + torch.log(beta_weights + eps)

        beta_log_prob = torch.logsumexp(weighted_beta_log_probs, dim=-1)
        return torch.stack([categorical_log_prob, beta_log_prob], dim=-1)

    def entropy(self, betfracs=None):
        """
        Compute (or approximate) the entropy of this unified distribution:
          - Categorical entropy (over action categories).
          - Plus, for the "raise" category (index = self.raise_category),
            the entropy of the Beta mixture distribution, weighted by
            the probability of choosing that category.
          - If `self.resistance` is used (meaning bet fractions are compressed),
            applies a correction for the transform's Jacobian.

        Args:
            betfracs (Tensor, optional): The fractional bet values actually sampled or used.
              Must be provided if `self.resistance` is not None, so we can invert the compression
              and compute the proper log-derivative correction.

        Returns:
            Tensor of shape (B, T) giving the approximate entropy for each batch/time entry.
        """

        eps = 1e-8

        # -------------------------------------------------------------
        # 1. CATEGORICAL ENTROPY
        # -------------------------------------------------------------
        # Shape of category_logits is (B, T, 5) typically
        categorical_dist = Categorical(logits=self.category_logits)
        # This returns a Tensor of shape (B, T)
        categorical_entropy = categorical_dist.entropy()

        # -------------------------------------------------------------
        # 2. BETA MIXTURE ENTROPY (for the "raise" category)
        # -------------------------------------------------------------
        # (a) Entropy of each Beta component
        #     self.beta_components.entropy() has shape (B, T, num_betas)
        beta_entropies = self.beta_components.entropy()  # (B, T, num_betas)

        # (b) Mixture weights
        beta_weights = F.softmax(self.beta_logits, dim=-1)  # (B, T, num_betas)

        # (c) Weighted sum of component entropies
        weighted_beta_entropy = torch.sum(beta_weights * beta_entropies, dim=-1)  # (B, T)

        # (d) Entropy of the mixture weights themselves
        mixture_entropy = -torch.sum(beta_weights * torch.log(beta_weights + eps), dim=-1)  # (B, T)

        # (e) Approx total Beta mixture entropy
        #     (Note: this is an upper bound for mixture entropy in general,
        #      ignoring overlap among components; commonly used in RL.)
        beta_entropy = weighted_beta_entropy + mixture_entropy  # (B, T)


        # -------------------------------------------------------------
        # 3. COMBINE WITH THE PROBABILITY OF (LAST) RAISE CATEGORY
        # -------------------------------------------------------------
        # The final distribution is:
        #   P(category < raise_category) for discrete,
        #   P(category=raise_category) * BetaMixture(...) for continuous.
        # A common RL approach is to do:
        #   total_entropy = H(categorical) + p(raise) * H(BetaMixture)
        #
        # category_probs has shape (B, T, 5), so we pick the last category's probability:
        p_raise = self.category_probs[:, :, -1]  # shape (B, T)
        combined_entropy = torch.stack([categorical_entropy, p_raise * beta_entropy], axis=-1)

        return combined_entropy


class MixedDistributionHead(torch.nn.Module):
    def __init__(
            self,
            input_size,
            layers_cat=(128,),
            layers_cont=(128,),
            activation=torch.nn.GELU,
            num_components=3,

    ):
        super().__init__()
        self.linear_layers_cat = self._generate_linear_layers(input_size, layers_cat)
        self.fc_cat = torch.nn.Linear(layers_cat[-1], 3)
        self.linear_layers_cont = self._generate_linear_layers(input_size, layers_cont)
        self.fc_cont = torch.nn.Linear(layers_cont[-1], num_components * 3)
        for layer in [self.fc_cat, self.fc_cont]:
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        self.activation = activation()
        self.num_components = num_components
        self._softplus = torch.nn.Softplus()

    @staticmethod
    def _generate_linear_layers(input_size, layers):
        layers_list = torch.nn.ModuleList()
        current_input_size = input_size
        for neurons in layers:
            layers_list.append(torch.nn.Linear(current_input_size, neurons))
            current_input_size = neurons

        for layer in layers_list:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        return layers_list
    @staticmethod
    def track_gradient(name):
        def hook(grad):
            global stop_execution
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"{name} has NaN or Inf gradients!")
            elif grad.abs().max() > 100:
                stop_execution.set()
                print(f"{name} has crazy large gradients! Max: {grad.abs().max().item()}")
            elif grad.abs().max() < 1e-6:
                print(f"{name} has tiny gradients! Max: {grad.abs().max().item()}")

        return hook

    def forward(self, x, action_mask=None):
        # categorical
        current_input = x
        for layer in self.linear_layers_cat:
            current_input = self.activation(layer(current_input))
        category_logits = self.fc_cat(current_input)

        # guassian

        current_input = x
        for layer in self.linear_layers_cont:
            current_input = self.activation(layer(current_input))
        x_mixture = self.fc_cont(current_input)
        a, b, logits = torch.chunk(x_mixture, self.num_components, dim=-1)
        a = self._softplus(a)
        b = self._softplus(b)
        # ✅ Register hooks ONLY if requires_grad is True
        if a.requires_grad:
            a.register_hook(self.track_gradient("a"))
        if b.requires_grad:
            b.register_hook(self.track_gradient("b"))
        if logits.requires_grad:
            logits.register_hook(self.track_gradient("logits"))

        return UnifiedDistribution(
            category_logits=category_logits,
            beta_alphas=a,
            beta_betas=b,
            beta_logits=logits,
            categorical_mask=action_mask,
        )



if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from pokerAI.policies.split_gru_policy import SplitGruPolicy

    hidden_size = 128
    input_size_recurrent = 42
    input_size_static = 30
    linear_layers = (256, 256)
    feature_dim = 259
    model =  SplitGruPolicy(
            input_size_recurrent,
            feature_dim + input_size_static +2,
            hidden_size,
            1,
            linear_layers,
        )
    path = "../policies/saved_models/best_model.pt"
    path_save = "../policies/saved_models/best_model_with_compression.pt"
    parameters = torch.load(path)
    new_parameters = model.state_dict()
    filtered_state_dict = {k: v for k, v in parameters.items() if k in new_parameters and "beta_layer" not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    torch.save(model.state_dict(), path_save)
    optimizer = torch.optim

