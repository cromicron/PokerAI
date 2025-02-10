from torch import nn
class GRUModule(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_gru_layers,
            linear_layers,
            activation=nn.LeakyReLU,
            output_dim=1,
            output_activation=None,
    ):
        """
        Initialize the GRU module with multiple layers in one GRU and Linear layers.

        Args:
            input_size (int): The size of the input features to the GRU.
            hidden_size (int): The number of neurons in each GRU layer.
            num_gru_layers (int): The number of GRU layers.
            linear_layers (tuple): A tuple where each element specifies the number of neurons in each Linear layer.
            activation (nn.Module): The non-linearity to apply after intermediate linear layers. Default is ReLU.
            output_dim (int): The size of the final output layer. Default is 1.
            output_activation (nn.Module or None): Optional activation for the final output layer.
        """
        super(GRUModule, self).__init__()

        # Define a single multi-layer GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True
        )

        # Define Linear layers
        self.linear_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.output_activation = output_activation

        # First linear layer input is hidden_size (GRU output)
        current_input_size = hidden_size

        for neurons in linear_layers:
            self.linear_layers.append(nn.Linear(current_input_size, neurons))
            if activation is not None:
                self.activations.append(activation())
            current_input_size = neurons  # Update input size for the next layer

        if output_dim:
            self.linear_layers.append(nn.Linear(current_input_size, output_dim))

    def forward(self, x, hidden_state=None, return_sequences=False):
        """
        Forward pass through the GRU and Linear layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            hidden_state (torch.Tensor or None): Hidden state for the GRU.
                                                 Shape: (num_layers, batch_size, hidden_size).
            return_sequences (bool): Whether to return outputs for all timesteps (True)
                                      or only the last timestep (False).

        Returns:
            torch.Tensor: Output tensor after passing through GRU and Linear layers.
                          Shape: (batch_size, seq_length, output_dim) if return_sequences=True,
                                 (batch_size, output_dim) if return_sequences=False.
            torch.Tensor: Hidden state of the GRU.
                          Shape: (num_layers, batch_size, hidden_size).
        """
        # Forward pass through the GRU
        x, hidden_state = self.gru(x, hidden_state)  # GRU handles all layers internally

        if return_sequences:
            # Process each timestep's output through the linear layers
            out = x  # Shape: (batch_size, seq_length, hidden_size)
            for layer, activation in zip(self.linear_layers, self.activations + [None]):
                out = layer(out)
                if activation:
                    out = activation(out)
        else:
            # Take the last time step's output for linear layers
            out = x[:, -1, :]  # Shape: (batch_size, hidden_size)
            for layer, activation in zip(self.linear_layers, self.activations + [None]):
                out = layer(out)
                if activation:
                    out = activation(out)
            out = out.unsqueeze(1)
        if self.output_activation:
            out = self.output_activation(out)
        return out, hidden_state
