from pokerAI.encoders.hand_encoder import PokerHandEmbedding
import numpy as np
import torch
import os

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "hand_encoder.pth")
embedding_dim = 4  # Size of individual card embeddings
# Example usage
feature_dim = 256  # Size of output feature vectors
deep_layer_dims = (512, 2048, 2048, 2048)
intermediary_dim = 16




class Encoder(PokerHandEmbedding):
    def __init__(self):
        super().__init__(embedding_dim, feature_dim, deep_layer_dims, intermediary_dim)
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu'))  # Load the checkpoint file
        self.load_state_dict(checkpoint["model_state_dict"])  # Load model weights
        self.to("cpu")
        self.eval()
        self = torch.compile(self)

    @torch.no_grad()
    def encode(self, holecards, flop=None, turn=None, river=None):
        """
        Encodes hole cards and community cards into features and probabilities.

        Args:
            holecards: Can be a 1D list of integers, a 2D NumPy array, or a 2D PyTorch tensor.
            flop: Can be a 1D list of integers or None.
            turn: Can be a single integer, a 1D NumPy array, or a 2D NumPy array.
            river: Can be a single integer, a 1D NumPy array, or a 2D NumPy array.

        Returns:
            A concatenated tensor of features and probabilities.
        """

        # Convert holecards to a 2D tensor if it's not already
        if isinstance(holecards, np.ndarray):
            holecards = torch.from_numpy(holecards)
        elif isinstance(holecards, list):
            holecards = torch.tensor(holecards)

        if holecards.dim() == 1:
            holecards = holecards.unsqueeze(dim=0)  # Ensure it's 2D

        # Convert flop to a 2D tensor if provided
        if flop is not None:
            if isinstance(flop, np.ndarray):
                flop = torch.from_numpy(flop)
            elif isinstance(flop, list):
                flop = torch.tensor(flop)
            if flop.dim() == 1:
                flop = flop.unsqueeze(dim=0)  # Ensure it's 2D

        # Convert turn to a 2D tensor if provided
        if turn is not None:
            if isinstance(turn, np.ndarray):
                turn = torch.from_numpy(turn)
            elif isinstance(turn, int):
                turn = torch.tensor([[turn]])  # Ensure it's 2D
            elif isinstance(turn, list):
                turn = torch.tensor(turn)
            if turn.dim() == 1:
                turn = turn.unsqueeze(dim=0)  # Ensure it's 2D

        # Convert river to a 2D tensor if provided
        if river is not None:
            if isinstance(river, np.ndarray):
                river = torch.from_numpy(river)
            elif isinstance(river, int):
                river = torch.tensor([[river]])  # Ensure it's 2D
            elif isinstance(river, list):
                river = torch.tensor(river)
            if river.dim() == 1:
                river = river.unsqueeze(dim=0)  # Ensure it's 2D

        try:
            # Pass inputs to the module
            features, log_probs = self(holecards, flop, turn, river)

            # Determine which probabilities to use based on the game stage
            if river is not None:
                probs = torch.exp(log_probs["log_probs_outcome_river"])
            elif turn is not None:
                probs = torch.exp(log_probs["log_probs_outcome_turn"])
            elif flop is not None:
                probs = torch.exp(log_probs["log_probs_outcome_flop"])
            else:
                probs = torch.exp(log_probs["log_probs_outcome_preflop"])

            # Concatenate features and probabilities
            return torch.cat([features, probs], dim=-1)

        except Exception as e:
            # Debug point
            print(f"Error: {e}")
            print(f"turn type: {type(turn)}, shape: {turn.shape if isinstance(turn, torch.Tensor) else 'N/A'}")
            print(f"river type: {type(river)}, shape: {river.shape if isinstance(river, torch.Tensor) else 'N/A'}")
            return None

    def to(self, *args, **kwargs):
        return self



