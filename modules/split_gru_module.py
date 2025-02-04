from torch import nn
import matplotlib.pyplot as plt
import torch


class SplitGRUModule(nn.Module):
    def __init__(
            self,
            input_size_recurrent,
            input_size_regular,
            hidden_size,
            num_gru_layers,
            linear_layers,
            activation=nn.GELU,
    ):
        super().__init__()
        self._input_size_recurrent = input_size_recurrent
        self._input_size_regular = input_size_regular
        # GRU Layer for processing betting sequences only
        self.gru = nn.GRU(
            input_size=input_size_recurrent,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True
        )
        # Reguluar Layers for processing stationary inputs
        # Define Linear layers
        self.linear_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        current_input_size = input_size_regular + hidden_size

        for neurons in linear_layers:
            self.linear_layers.append(nn.Linear(current_input_size, neurons))
            if activation is not None:
                self.activations.append(activation())
            current_input_size = neurons  # Update input size for the next layer

        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                # Apply Xavier (Glorot) Uniform Initialization to all weight matrices
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize biases to zero (optional, can be constant or other strategies too)
                torch.nn.init.zeros_(param)


    def forward(self, x, hidden_state=None, return_sequences=False):
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
        x_recurrent = x[...,: self._input_size_recurrent]
        x_static = x[..., self._input_size_recurrent: ]
        out_gru, hidden_state = self.gru(x_recurrent, hidden_state)  # GRU handles all layers internally

        if return_sequences:
            # Process each timestep's output through the linear layers
            out = torch.cat([x_static, out_gru], dim=-1)
            for layer, activation in zip(self.linear_layers, self.activations + [None]):
                out = layer(out)
                if activation:
                    out = activation(out)
        else:
            # Take the last time step's output for linear layers
            out = torch.cat([x_static[:, -1, :], out_gru[:, -1, :]], dim=-1)
            for layer, activation in zip(self.linear_layers, self.activations + [None]):
                out = layer(out)
                if activation:
                    out = activation(out)
            out = out.unsqueeze(1)

        return out, hidden_state
