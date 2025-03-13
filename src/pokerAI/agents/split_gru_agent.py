import torch

class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.legal_actions = []
        self.reward = None
        self.ranges = []
        self.action_probs = []

    def add_observation(self, observation):
        self.observations.append(observation)

    def add_action(self, action, legal_action):
        self.actions.append(action)
        self.legal_actions.append(legal_action)

    def finish_episode(self, reward):
        self.reward = reward

    def add_range(self, hand_range):
        self.ranges.append(hand_range)

    def add_action_prob(self, action_prob):
        self.action_probs.append(action_prob)




class Agent:
    def __init__(
            self,
            player,
            policy,
            value_function,
            encode_value=False,
            idx_stack_start=51,
            idx_stack_now=30,
            idx_bet=41,
            policy_path=None,
            value_path=None

    ):
        self.player = player
        self.policy = policy
        if encode_value:
            assert value_function is not None, "you must provide a value function if you want to encode value"
            self.idx_stack_start=idx_stack_start,
            self.idx_stack_now=idx_stack_now,
            self.idx_bet=idx_bet,
        self.value_function = value_function
        if policy_path:
            self.policy.load_state_dict(torch.load(policy_path))
        if value_path:
            self.value_function.load_state_dict(torch.load(value_path))
        self.episodes = []
        self.create_episode()

    def encode_state(self, state, range=False):
        return self.policy.encode_state(state, range)

    def get_action(self, game, smallest_unit=1, temperature=1, play_range=False, update_range=False):
        return self.policy.get_action(
            game, smallest_unit, temperature=temperature, play_range=play_range, update_range=update_range)


    def add_to_sequence(self, state):
        self.policy.add_to_sequence(state)

    def add_to_sequence_range(self, state):
        self.policy.add_to_sequence_range(state)

    def clear_episode_buffer(self):
        del self.episodes
        self.episodes = []

    def create_episode(self):
        self.episodes.append(Episode())

    def reset(self):
        self.clear_episode_buffer()
        self.player.stack = self.player.starting_stack
        self.blind=0
        self.policy.reset()

