from encoders.embedding_hand_encoder import encode
from PokerGame.NLHoldem import Game, Card, Player
from value_functions.split_gru_value_function import GRUValueFunction
from policies.split_gru_policy import SplitGruPolicy
from lookup.HandComparatorLookup import strength_array, strength
import torch
import numpy as np
from algos.ppo import Agent, Episode, check_hand_evals
from typing import List
import pandas as pd
import gc
from tqdm import tqdm
import os
from copy import deepcopy
import ray
from time import time

torch.set_flush_denormal(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["RAY_memory_usage_threshold"] = ".98"
ray.init()


class LeaguePlayer(Agent):
    def __init__(
            self,
            player,
            policy,
            value_function,
            encode_value=False,
            name="default",
            path_policy_network=None,
            path_value_network=None,
            path_policy_optimizer=None,
            path_value_optimizer=None,
            load=False,
    ):
        super().__init__(player, policy, value_function, encode_value)
        self.n_games = 0
        self.score = 0
        self.name = name

        self._paths = {
            "policy_network": path_policy_network,
            "value_network": path_value_network,
            "policy_optimizer": path_policy_optimizer,
            "value_optimizer": path_value_optimizer
        }


        if load:
            with torch.no_grad():
                self.policy.load_state_dict(torch.load(self._paths["policy_network"], map_location="cpu", weights_only=True))
                self.policy.to("cpu")  # Ensure policy is explicitly on CPU
                self.policy.module.gru.gru.flatten_parameters()
                self.policy.eval()

                self.value_function.load_state_dict(torch.load(self._paths["value_network"], map_location="cpu", weights_only=True))
                self.policy_optimizer.load_state_dict(torch.load(self._paths["policy_optimizer"], map_location=device, weights_only=True))
                self.value_optimizer.load_state_dict(torch.load(self._paths["value_optimizer"], map_location=device, weights_only=True))

        for optimizer in [self.policy_optimizer, self.value_optimizer]:
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)

    def change_player(self, player):
        self.player = player

    def save_model(self):
        torch.save(self.policy.state_dict(), self._paths["policy_network"])
        torch.save(self.policy_optimizer.state_dict(), self._paths["policy_optimizer"])
        torch.save(self.value_function.state_dict(), self._paths["value_network"])
        torch.save(self.value_optimizer.state_dict(), self._paths["value_optimizer"])

class Match:
    def __init__(self, player_1: LeaguePlayer, player_2: LeaguePlayer, max_games=400):
        self.n_games = 0
        self.players = [player_1, player_2]
        self.max_games = max_games
        self.finished = False
        self.result = None
    def play_match(self, reward_type = "ranges_only"):
        starting_scores = {agent: agent.score for agent in self.players}
        assert self.finished == False, "This game is already finished"
        game = Game(stacks=[agent.player.starting_stack for agent in self.players])
        if reward_type != "regular":
            assert game.n_players == 2, "This feature is only implemented for two-player games."
        agents_dict = {}
        for player, agent in zip(game.players, self.players):
            agent.change_player(player)
            agents_dict[player] = agent
        game.new_hand(first_hand=True)
        while True:
            # Start a new hand and create an Episode object for each player

            for agent in self.players:
                # check blinds
                agent.blind = agent.player.bet
                agent.episodes.append(Episode())
                agent.policy.reset()
            street = 0
            while not game.finished:
                # Record observations for all players at this step
                for agent in self.players:
                    state = game.get_state(agent.player)
                    recurrent_state, static_state = agent.policy.encode_state(state)
                    hole_int = agent.player.hole_int
                    if game.street >0:
                        flop_int, turn_int, river_int = game.board_int
                        if game.street > street:
                            if street == 0:
                                cards_to_remove = flop_int
                            elif street == 1:
                                cards_to_remove = [turn_int]
                            else:
                                cards_to_remove = [river_int]
                            agent.policy.remove_cards_from_range(cards_to_remove)
                            agent.policy.update_feature_array(flop_int, turn_int, river_int)

                    else:
                        flop_int = turn_int = river_int = None
                    features = encode(hole_int, flop_int, turn_int, river_int)

                    if features is None:
                        a = 1
                        print("here")
                    features = torch.cat([
                        recurrent_state,
                        static_state,
                        features.squeeze(),
                    ], dim=-1)
                    agent.episodes[-1].add_observation(features)
                    if agent.player != game.acting_player:
                        agent.episodes[-1].add_action(None, None)
                    agent.add_to_sequence(features)
                    try:
                        agent.add_to_sequence_range(features[:recurrent_state.shape[0]+static_state.shape[0]])
                    except:
                        a = 1
                        print(a)

                # Acting player decides an action
                street = game.street
                current_player = game.acting_player
                state = game.get_state(current_player)
                action, betsize, action_type, betfrac, legal_actions = agents_dict[current_player].get_action(state, temperature=exploration_temp)
                action_add = (action_type, betfrac)
                agents_dict[current_player].episodes[-1].add_action(action_add, legal_actions)

                game.implement_action(current_player, action, betsize)


            # At the end of the hand, record the rewards for all players
            # If
            if (len(game.left_in_hand) > 1) and (reward_type in ("single_range", "ranges_only")):
                # there was a showdown and thus each player has it's own range
                board = [(card.value, card.suit) for card in game.board]
                board_array = np.tile(board, (1326, 1, 1))
                board_int = []
                for street in game.board_int:
                    if type(street) in (list, tuple):
                        board_int.extend(street)
                    else:
                        board_int.append(street)
            for agent in self.players:
                if (len(game.left_in_hand) == 1) or (reward_type == "regular"):
                    score = agent.player.stack - agent.player.starting_stack
                elif reward_type in ("single_range", "ranges_only"):
                    if reward_type == "single_raise":
                        all_cards_to_remove = board_int + agent.player.hole_int
                    else:
                        all_cards_to_remove = board_int
                        agent.policy.remove_cards_from_range(all_cards_to_remove)
                    # remove all board cards and self-cards from range
                    villain = [p.policy for p in self.players if p != agent][0]
                    villain_range = villain.range.copy()
                    villain.remove_cards_from_range(all_cards_to_remove)

                    villain_all_hands = np.concatenate([villain.all_holecards_tuple, board_array], axis=1)
                    villain_strengths = strength_array(villain_all_hands)
                    if reward_type == "single_raise":
                        remaining_hands = villain.range != 0
                        villain_strengths = villain_strengths[remaining_hands]
                        hero_cards = [(card.value, card.suit) for card in agent.player.holecards] + board
                        hero_strength = strength(hero_cards)
                        p_win = ((hero_strength > villain_strengths) * villain.range[remaining_hands]).sum()
                        p_tie = ((hero_strength == villain_strengths) * villain.range[remaining_hands]).sum()

                    else:
                        hero_range = agent.policy.range.copy()
                        valid_mask = agent.policy.valid_comparisons

                        hero_all_hands = np.concatenate([agent.policy.all_holecards_tuple, board_array], axis=1)
                        hero_strengths = strength_array(hero_all_hands)


                        villain_range_probs =  villain.range[None, :]
                        win_weighted = np.sum(
                            (
                                    hero_strengths[:, None] > villain_strengths
                            )* villain_range_probs *valid_mask, axis=1)
                        tie_weighted = np.sum(
                            (
                                    hero_strengths[:, None] == villain_strengths
                            ) * villain_range_probs * valid_mask, axis=1)

                        total_weights = np.sum(valid_mask * villain_range_probs, axis=1)

                        p_win = win_weighted / total_weights
                        p_tie = tie_weighted / total_weights

                        agent.policy.range = hero_range

                    score = agent.player.stack_before_showdown + p_win * game.pot + 1 / (
                        len(game.left_in_hand)) * p_tie * game.pot - agent.player.starting_stack
                    villain.range = villain_range

                reward = score + agent.blind
                agent.episodes[-1].finish_episode(reward)
                if reward_type == "ranges_only":
                    agent.score += (score * agent.policy.range).sum()
                else:
                    agent.score += score
                agent.n_games += 1
            self.n_games += 1
            if self.n_games == self.max_games:
                break
            else:
                game.new_hand(first_hand=False)
        self.result = {agent: agent.score - starting_scores[agent] for agent in starting_scores}
        self.finished = True
        game.new_hand(first_hand=False)




class Matchday:
    def __init__(self, pairings: list):
        self.pairings = pairings

    def play_matchday(self):
        for pairing in self.pairings:
            pairing.play_match()


@ray.remote
def play_matchday_remote(matchday: Matchday):
    matchday.play_matchday()
    return matchday.pairings
    for match in matchday.pairings:
        for player in match.players:
            player.policy = None
            player.value_function = None
    return matchday.pairings

class League:
    def __init__(
            self,
            players: List[LeaguePlayer],
            games_per_match=500,
            games_till_evolution=100_000,
            games_per_match_evolution=10,
            n_evolutions=1,
            path_evolutions_pol= "../policies/saved_models/evolutions/",
            path_evolutions_val="../value_functions/saved_models/evolutions/",
    ):
        assert len(players) % 2 == 0, "Provide an even number of teams"
        self.players = players
        self.games_per_match = games_per_match
        self.schedule = None
        self.create_schedule()
        self.games_till_evolution = games_till_evolution
        self.games_per_match_evolution = games_per_match_evolution
        self.n_evolutions = n_evolutions
        self.path_evolution_policy = path_evolutions_pol
        self.path_evolution_value = path_evolutions_val
    def create_schedule(self, games_per_match=None):
        """
        Generate a round-robin schedule for a list of teams.

        Returns:
            list of Matchday: Each Matchday contains the matches (pairs of teams) for that day.
        """
        if games_per_match is None:
            games_per_match = self.games_per_match
        teams = self.players.copy()  # Make a copy to avoid modifying the original list
        num_teams = len(teams)
        num_rounds = num_teams - 1
        schedule = []

        # Rotate the list of teams while keeping the first team fixed
        for round_idx in range(num_rounds):
            pairings = []
            for i in range(num_teams // 2):
                team1 = teams[i]
                team2 = teams[-(i + 1)]
                pairings.append(Match(team1, team2, max_games=games_per_match))
            schedule.append(Matchday(pairings))
            # Rotate teams (except the first one)
            teams = [teams[0]] + teams[-1:] + teams[1:-1]

        self.schedule=schedule

    def evolution(self, remote=True):
        """play longer league without training and establish worst player"""
        # save legacy
        new_path_policies = f"{self.path_evolution_policy}/{str(self.n_evolutions).zfill(3)}"
        new_path_value = f"{self.path_evolution_value}/{str(self.n_evolutions).zfill(3)}"
        os.makedirs(new_path_policies, exist_ok=True)
        os.makedirs(new_path_value, exist_ok=True)
        for a_nr, agent in enumerate(self.players):
            torch.save(agent.policy.state_dict(), os.path.join(new_path_policies, f"policy_{a_nr}.pt"))
            torch.save(agent.policy_optimizer.state_dict(), os.path.join(new_path_policies, f"optimizer_{a_nr}.pt"))
            torch.save(agent.value_function.state_dict(), os.path.join(new_path_value, f"model_{a_nr}.pt"))
            torch.save(agent.value_optimizer.state_dict(), os.path.join(new_path_value, f"optimizer_{a_nr}.pt"))
            agent.reset()
            agent.score = 0
            gc.collect()
            agent.value_function.to("cpu")
        self.create_schedule(self.games_per_match_evolution)
        self.play_season(remote)
        result =  sorted(self.players, key=lambda x: x.score, reverse=True)
        print(self.table)
        # replace policy, value_function, and optimizers of worst with best players
        for i in range(2):
            # Get the indices
            idx = -i - 1

            # Deep copy policies and value functions
            result[idx].policy = deepcopy(result[i].policy)
            result[idx].value_function = deepcopy(result[i].value_function)

            # Reinitialize optimizers to match the new objects
            result[idx].policy_optimizer = type(result[i].policy_optimizer)(
                result[idx].policy.parameters(),
                **result[i].policy_optimizer.defaults
            )
            result[idx].value_optimizer = type(result[i].value_optimizer)(
                result[idx].value_function.parameters(),
                **result[i].value_optimizer.defaults
            )
        for agent in self.players:
            agent.score = 0
            if remote:
                agent.n_games -= self.games_per_match * (len(self.players) -1)
            agent.clear_episode_buffer()
            agent.policy.to("cpu")
            agent.value_function.to("cpu")
        self.n_evolutions += 1

    def play_season(self, remote=True):
        """Play all matches in the season"""
        if remote:
            self._detach_optimizers()
            # keep episodes and don't pass to remote
            player_dict = {player.name: player for player in self.players}
            futures = [play_matchday_remote.remote(matchday) for matchday in self.schedule]
            results = ray.get(futures)
            for matchday in results:
                for match in matchday:
                    for sim_player in match.players:
                        name = sim_player.name
                        player_dict[name].episodes.extend(sim_player.episodes)
                        player_dict[name].score += match.result[sim_player]
                        player_dict[name].n_games += match.n_games
            self._attach_optimizers()
        else:
            for matchday in tqdm(self.schedule):
                 matchday.play_matchday()

    @property
    def table(self):
        # Collect player data into a list of dictionaries
        data = [
            {
                "Name": player.name,
                "Score": player.score,
                "Games Played": player.n_games,
            }
            for player in sorted(self.players, key=lambda x: x.score, reverse=True)  # Sort by score descending
        ]
        # Create a pandas DataFrame
        return pd.DataFrame(data)

    def _detach_optimizers(self):
        """temporarily remove optimizers from players"""
        self.optimizers = {}
        for player in self.players:
            self.optimizers[player] = (player.value_optimizer, player.policy_optimizer)
            player.value_optimizer = None
            player.policy_optimizer = None

    def _attach_optimizers(self):
        """temporarily remove optimizers from players"""
        for player in self.players:
            player.value_optimizer, player.policy_optimizer = self.optimizers[player]


num_agents = 8
hands_per_round = 500
players_per_hand = 2
stack = 20
exploration_temp = 1.1
entropy_coeff = 1

path_policies = "../policies/saved_models/"
path_value_functions = "../value_functions/saved_models/"
load_for_model = [False for _ in range(num_agents)]


game = Game(n_players=2, stacks=[stack for _ in range(players_per_hand)])
game.new_hand(first_hand=True)

# Compute state vector size using a dummy state

hidden_size = 128
input_size_recurrent = 42
input_size_static = 30
linear_layers = (256, 256)
feature_dim = 259
value_functions = [
    GRUValueFunction(
            input_size_recurrent,
            feature_dim + input_size_static,
            hidden_size,
            1,
            linear_layers=(256, 128),
        ) for _ in range(num_agents)
]

policies = [
    SplitGruPolicy(
        input_size_recurrent,
        feature_dim + input_size_static +2,
        hidden_size,
        1,
        linear_layers,
        value_function=v,
    ) for v in value_functions
]

league = League([
    LeaguePlayer(
        Player(i, stack),
        policy,
        value_function,
        True,
        str(i),
        path_policy_network=f"{path_policies}policy_{i}.pt",
        path_value_network=f"{path_value_functions}model_{i}.pt",
        path_policy_optimizer=f"{path_policies}optimizer_{i}.pt",
        path_value_optimizer=f"{path_value_functions}optimizer_{i}.pt",
        load=load,

    ) for i, (policy, value_function, load) in enumerate(zip(policies, value_functions, load_for_model))
], games_per_match=hands_per_round, n_evolutions=0)

while True:
    for a_nr, agent in enumerate(league.players):
        agent.reset()
        gc.collect()
        agent.policy.to("cpu")
        agent.value_function.to("cpu")
        #check_hand_evals(agent, output_file=f"evaluations/hand_evaluations_{a_nr}.csv", new_game=game)
    if agent.n_games >= league.n_evolutions*league.games_till_evolution:
        print("Evolving Players")
        league.evolution(False)

    league.create_schedule()
    now = time()
    league.play_season()
    print(f"finished playing in {time() - now}")
    print(league.table)
    remote=True
    for agent in league.players:
        agent.generate_training_data()
        agent.train_policy(epochs=3, batch_size=64, entropy_coef=entropy_coeff, clip_epsilon=0.05, verbose=False)
        agent.policy.to("cpu")
        agent.value_function.to("cpu")
        with torch.no_grad():
            state_input = agent.training_data["observations"]
            value_predictions = agent.value_function(state_input, return_sequences=True)[0]
            state_input = torch.cat([state_input, value_predictions], dim=-1)
            probs = agent.policy(state_input, return_sequences=True)[0].category_probs
        action_types = agent.training_data["action_types"]
        action_masks = agent.training_data["action_masks"]
        agent.value_function.train_on_data(
            optimizer=agent.value_optimizer,
            epochs=3,
            observations=agent.training_data["observations"],
            rewards=agent.training_data["rewards"],
            mask=agent.training_data["masks"],
            actions=action_types,
            action_masks=action_masks,
            probs=probs,
            verbose=False,
        )
        agent.save_model()
