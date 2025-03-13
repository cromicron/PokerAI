import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from PokerGame.NLHoldem import Game, Card, Player
from pokerAI.policies.split_gru_policy import SplitGruPolicy
from pokerAI.algos.ppo import check_hand_evals
import torch
import numpy as np
from pathlib import Path
from heads_up_competition import LeaguePlayer, Match
import pandas as pd
from copy import deepcopy
import ray
import gc


SCRIPT_DIR = Path(__file__).parent
torch.set_flush_denormal(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["RAY_memory_usage_threshold"] = ".99"


def show_ray_usage():
    resources = ray.available_resources()

    # Convert bytes to GB for memory-related values
    def format_memory(bytes_value):
        return f"{bytes_value / (1024 ** 3):.2f} GB"

    # Convert and print the resources in a readable way
    print("\n📊 **Ray Available Resources** 📊")
    for key, value in resources.items():
        if "memory" in key:
            print(f"🖥️ {key}: {format_memory(value)}")
        else:
            print(f"⚙️ {key}: {value}")

@ray.remote
def play_match_parallel(learner, villain, max_games, store_episodes=None):
    match = Match(deepcopy(learner), deepcopy(villain), max_games=max_games)
    match.play_match(store_episodes=store_episodes)
    return match  # Optional: Return match results if needed

def play_match(learner, villain, max_games, store_episodes=None):
    match = Match(deepcopy(learner), deepcopy(villain), max_games=max_games)
    match.play_match(store_episodes=store_episodes)
    return match  # Optional: Return match results if needed

def play_round(player, villains, max_games=10, parallel=True, store_episodes=None):
    if store_episodes is None:
        store_episodes = [player.name]
    policy_optimizer = player.policy_optimizer
    value_optimizer = player.value_optimizer
    episodes = player.episodes
    player.episodes = []
    player.policy_optimizer, player.value_optimizer = None, None
    hero_score_start = player.score
    hero_n_game_start = player.n_games
    player.score = 0
    player.n_games = 0
    if parallel:
        futures = [play_match_parallel.remote(player, villain, max_games, store_episodes) for villain in villains]
        results = ray.get(futures)
    else:
        results = [play_match(player, villain, max_games, store_episodes) for villain in villains]
    player.policy_optimizer = policy_optimizer
    player.value_optimizer = value_optimizer
    games_played = 0
    scores_round = 0

    for i in range(len(results)):
        hero = results[i].players[0]
        games_played += hero.n_games
        scores_round += hero.score
        villain = results[i].players[1]
        villains[i].score += villain.score - villains[i].score
        villains[i].n_games += villain.n_games - villains[i].n_games
        episodes.extend(hero.episodes)
    player.episodes = episodes
    player.score = hero_score_start + scores_round
    player.n_games = hero_n_game_start + games_played
    del(results)
    gc.collect()

def get_scores(villains, learner=None, expert=None):
    players = [v for v in villains]
    if learner is not None:
        players.append(learner)
    if expert is not None:
        players.append(expert)
    data = [
        {
            "Name": player.name,
            "Score": player.score,
            "Games Played": player.n_games,
        }
        for player in sorted(players, key=lambda x: x.score, reverse=True)  # Sort by score descending
    ]
    # Create a pandas DataFrame
    return pd.DataFrame(data)

class ExpertCopy(LeaguePlayer):
    def __init__(
            self,
            player,
            policy,
            encode_value=True,

    ):
        super().__init__(
            player,
            policy,
            encode_value=encode_value,
            load=False,
            optimizer=False,
        )

    def change_policy(self, policy, value):
        self.policy = deepcopy(policy)
        self.value_function = deepcopy(value)



path_policies = (SCRIPT_DIR / "../src/pokerAI/policies/saved_models").resolve()
path_value_functions = (SCRIPT_DIR / "../src/pokerAI/value_functions/saved_models").resolve()


hidden_size = 128
input_size_recurrent = 43
input_size_static = 30
linear_layers = (256, 256)
feature_dim = 259
epochs_learner_value = 4
epochs_learner_policy = 4
epochs_expert_value = 4
epochs_expert_policy = 4
use_expert = True

policy_kwargs = {
    "input_size_recurrent": input_size_recurrent,
    "input_size_regular": feature_dim + input_size_static +2,
    "hidden_size": hidden_size,
    "num_gru_layers": 1,
    "linear_layers": linear_layers,
}
stack = 50
learner = LeaguePlayer(
    Player("learner", 50),
    SplitGruPolicy(**policy_kwargs),
    path_policy_network=f"{path_policies}/policy_learner.pt",
    path_value_network=f"{path_value_functions}/model_learner.pt",
    path_policy_optimizer=f"{path_policies}/optimizer_learner.pt",
    path_value_optimizer=f"{path_value_functions}/optimizer_learner.pt",
    load=True,
    exploration_temp=1.0,
)
check_hand_evals(learner, output_file=f"evaluations/hand_evaluations_learner.csv", new_game=Game(2, [stack, stack]),plot_densities=True)
if use_expert:
    expert = LeaguePlayer(
        Player("expert", 50),
        SplitGruPolicy(**policy_kwargs),
        path_policy_network=f"{path_policies}/policy_expert.pt",
        path_value_network=f"{path_value_functions}/model_expert.pt",
        path_policy_optimizer=f"{path_policies}/optimizer_expert.pt",
        path_value_optimizer=f"{path_value_functions}/optimizer_expert.pt",
        load=True,
        exploration_temp=1.0,
    )

    expert_copy = ExpertCopy(
        Player("expert_copy", 50),
        SplitGruPolicy(**policy_kwargs),
    )
    expert_copy.change_policy(expert.policy, expert.value_function)
n_processes = 6
n_processes_bust = 4
n_players = 20
max_games_expert = 80
max_games_regular = 20
interval_remove_policy = 2
interval_remove_policy_max = 20
n_remove = 2
n_games_remove = 25

def inverse_exponential_growth(epoch, start=interval_remove_policy, end=interval_remove_policy_max, growth_rate=0.005):
    return start + (end - start) * (1 - np.exp(-growth_rate * epoch))




player_pool = [LeaguePlayer(
    Player(f"player_{i}", stack),
    SplitGruPolicy(**policy_kwargs),
    load=True,
    path_policy_network=f"{path_policies}/policy_{i}.pt",
    path_value_network=f"{path_value_functions}/model_{i}.pt",
    path_policy_optimizer=f"{path_policies}/optimizer_{i}.pt",
    path_value_optimizer=f"{path_value_functions}/optimizer_{i}.pt",
    exploration_temp=1,
    optimizer=False
) for i in range(n_players)]
for player in player_pool:
    player.save_model()
ray.init()

show_ray_usage()
it = 39
while True:
    print("Epoch " + str(it))
# train expert
# deatch learner optimizer and episodes
    if use_expert:
        policy_optimizer = learner.policy_optimizer
        value_optimizer = learner.value_optimizer
        episodes = learner.episodes
        learner.episodes = []
        learner_score = learner.score
        learner_n_games = learner.n_games
        learner.policy_optimizer, learner.value_optimizer = None, None
        play_round(expert, [learner for _ in range(n_processes)], max_games_expert//n_processes, parallel=True)

        learner.policy_optimizer = policy_optimizer
        learner.value_optimizer = value_optimizer
        learner.episodes = episodes
        learner.score = learner_score
        learner.n_games = learner_n_games

        expert.generate_training_data()
        torch.cuda.empty_cache()
        expert.train_policy(
            epochs=epochs_expert_policy,
            batch_size=8192,
            entropy_coef_cat=0.2,
            entropy_coef_size=0.05,
            clip_epsilon_cat=0.1,
            clip_epsilon_cont=0.1,
            verbose=False,
            ranges=True
        )
        torch.cuda.empty_cache()
        expert.train_value_function(epochs=epochs_expert_value)

        torch.cuda.empty_cache()
        expert.save_model()
        expert.reset()
        #check_hand_evals(expert, output_file=f"evaluations/hand_evaluations_expert.csv", new_game=Game(2, [stack, stack]))
        #continue
        expert_copy.change_policy(expert.policy, expert.value_function)
        opponents = list(np.random.choice(player_pool, size=n_processes - 1, replace=False)) + [expert_copy]
    else:
        opponents = list(np.random.choice(player_pool, size=n_processes, replace=False))
    play_round(learner, opponents, max_games_regular, parallel=True)

    learner.generate_training_data()
    torch.cuda.empty_cache()
    learner.train_value_function(epochs=epochs_learner_value)
    torch.cuda.empty_cache()
    learner.train_policy(
        epochs=epochs_learner_policy,
        batch_size=8192,
        entropy_coef_cat=0.2,
        entropy_coef_size=0.05,
        clip_epsilon_cat=0.1,
        clip_epsilon_cont=0.1,
        verbose=False,
        ranges=True
    )
    torch.cuda.empty_cache()
    learner.reset()
    learner.save_model()
    if use_expert:
        print(get_scores(opponents,learner, expert))
    else:
        print(get_scores(opponents, learner))
    for villain in opponents:
        villain.reset()

    check_hand_evals(learner, output_file=f"evaluations/hand_evaluations_learner.csv", new_game=Game(2, [stack, stack]),plot_densities=True)
    if use_expert:
        check_hand_evals(expert, output_file=f"evaluations/hand_evaluations_expert.csv", new_game=Game(2, [stack, stack]),plot_densities=True)

    if it%interval_remove_policy == 0:
        # find worst players
        print("removing player")
        splits = len(player_pool)//n_processes_bust
        remain = len(player_pool) - splits*n_processes_bust
        for player in player_pool:
            player.score = 0
            player.n_games = 0
            player.episodes = []
        for i in range(splits):
            opponents = player_pool[i*n_processes_bust:(i+1)*n_processes_bust]
            play_round(learner, opponents, n_games_remove, parallel=True, store_episodes=[])
            learner.reset()
        if remain !=0:
            opponents = player_pool[splits*n_processes_bust:]
            play_round(learner, opponents, n_games_remove, parallel=True, store_episodes=[])
            learner.reset()

        scores = get_scores(player_pool)
        print(scores)
        worst_players = list(scores.tail(n_remove)["Name"].unique())
        last_include = "expert"
        for player in player_pool:
            if (player.name in worst_players) and (player.score < 0):
                if last_include == "expert":
                    replace_with = learner
                    last_include = "learner"
                else:
                    replace_with = expert
                    last_include = "expert"
                player.policy.load_state_dict(replace_with.policy.state_dict())
                player.value_function.load_state_dict(replace_with.value_function.state_dict())
                player.score = 0
                player.n_games = 0
                player.save_model()
            player.reset()

    it += 1
    interval_remove_policy = int(inverse_exponential_growth(it, interval_remove_policy))





