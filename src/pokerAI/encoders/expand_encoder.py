from hand_encoder import PokerHandEmbedding
import torch
import torch.nn as nn


def expand_linear_layer(old_layer: nn.Linear, new_out_features: int) -> nn.Linear:
    """Expands a Linear layer to new_out_features while keeping existing weights."""
    assert new_out_features >= old_layer.out_features, "New size must be larger or equal to old size"

    new_layer = nn.Linear(old_layer.in_features, new_out_features)

    # Copy existing weights into the new layer
    new_layer.weight.data[:old_layer.out_features, :] = old_layer.weight.data
    new_layer.bias.data[:old_layer.out_features] = old_layer.bias.data

    return new_layer


def expand_model_head(model: nn.Module, new_layer_dims: list):
    """
    Expands model.head_outcome_probs_river to the specified architecture in new_layer_dims,
    while preserving existing weights where possible.

    Args:
        model (nn.Module): The model whose head needs expansion.
        new_layer_dims (list): List of new hidden layer sizes. The last value must match the original output size.
                               Example: [128, 256, 3] expands to two hidden layers of sizes 128 and 256 before output.
    """
    old_layers = list(model.head_outcome_probs_river.children())  # Extract existing layers
    assert isinstance(old_layers[-1], nn.Linear), "Last layer must be a Linear layer (output layer)."

    old_output_layer = old_layers[-1]
    old_hidden_layers = old_layers[:-1]

    assert new_layer_dims[-1] == old_output_layer.out_features, "Output size must remain unchanged."

    new_layers = []
    input_dim = old_hidden_layers[0].in_features  # Get input feature dimension

    # Expand hidden layers dynamically
    for i, new_dim in enumerate(new_layer_dims[:-1]):  # Exclude last layer (output)
        if i < len(old_hidden_layers) and isinstance(old_hidden_layers[i], nn.Linear):
            old_layer = old_hidden_layers[i]
            assert new_dim >= old_layer.out_features, "New layer size must be >= old size."
            new_layer = nn.Linear(input_dim, new_dim)

            # Copy existing weights
            new_layer.weight.data[:old_layer.out_features, :] = old_layer.weight.data
            new_layer.bias.data[:old_layer.out_features] = old_layer.bias.data
        else:
            # If this layer didn’t exist in the original model, initialize from scratch
            new_layer = nn.Linear(input_dim, new_dim)

        new_layers.append(new_layer)
        new_layers.append(nn.GELU())  # Keep activation consistent
        input_dim = new_dim  # Update input for next layer

    # Output layer (unchanged)
    new_output_layer = nn.Linear(input_dim, old_output_layer.out_features)
    new_output_layer.weight.data[:, : old_output_layer.in_features] = old_output_layer.weight.data
    new_output_layer.bias.data = old_output_layer.bias.data
    new_layers.append(new_output_layer)

    # Replace model's head with the expanded one
    model.head_outcome_probs_river = nn.Sequential(*new_layers)




if __name__ == "__main__":
    MODEL_LOAD_PATH = "../training_scripts/best_model.pth"
    embedding_dim = 4  # Size of individual card embeddings
    # Example usage
    feature_dim = 256  # Size of output feature vectors
    deep_layer_dims = (512, 2048, 2048, 2048)
    intermediary_dim = 16

    parameters = torch.load(MODEL_LOAD_PATH)["model_state_dict"]
    model = PokerHandEmbedding(embedding_dim, feature_dim, deep_layer_dims, intermediary_dim)
    model.load_state_dict(parameters)
    expand_model_head(model, new_layer_dims=[256, 128, 64, 3])
    torch.save(model.state_dict(), "expanded_model_weights.pth")
