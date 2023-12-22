from tensorflow.keras.models import load_model
from tqdm import tqdm
from NLHoldem import Game
from Agents.ActorCritic import Agent

agent_0 = Agent()
agent_1 = Agent()
agent_0.actor = load_model("saved_models/actor_0")
agent_1.actor = load_model("saved_models/actor_1")

stacks = [20, 20]
game = Game(n_players=2, stacks=stacks, hand_history=True)
game.new_hand(first_hand=True)
map_agent = {game.players[0]: agent_0, game.players[1]: agent_1}

score_0 = 0
score_1 = 0

for k in tqdm(range(5000)):
    hand_0 = game.players[0].holecards
    hand_1 = game.players[1].holecards
    map_cards = {agent_0: hand_0, agent_1: hand_1}
    while not game.finished:
        next = game.next[-1]
        agent_to_act = map_agent[next]
        cards = map_cards[agent_to_act]
        hand_history = game.hand_history.hand_history
        position = game.positions.index(next)
        if position == 0:
            legal_actions = [0, 2]
        else:
            legal_actions = [0, 1]
        # legal_actions = game.get_legal_actions()
        minbet = game.get_legal_betsize()
        action, bet_size, bet_frac = agent_to_act.choose_action(cards, position, legal_actions, minbet, next.stack,
                                                                next.bet)
        if action == 2:
            game.implement_action(next, action, 20)
        else:
            game.implement_action(next, action)


    reward_0 = game.players[0].stack - game.players[0].starting_stack
    reward_1 = game.players[1].stack - game.players[1].starting_stack
    score_0 += reward_0
    score_1 += reward_1
    game.new_hand()

print("agent_0", score_0/k, "agent_1", score_1/k)
