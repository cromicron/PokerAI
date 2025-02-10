import abc
import torch

class Policy(torch.nn.Module, abc.ABC):
    def __init__(self, input_dims):
        super().__init__()
        self.input_dims = input_dims

    @abc.abstractmethod
    def get_action(self, observation, **kwargs):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.
            kwargs: additional keyword arguments

        Returns:
            Tuple[np.ndarray, dict[str,np.ndarray]]: Action and extra agent
                info.

        """