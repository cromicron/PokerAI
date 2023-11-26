import numpy as np
import pandas as pd
import tqdm
from NLHoldem import Game, Card
from Agents.ActorCritic import Agent

def generate_card_combinations():
    values = list(range(2, 15))  # 2 to 14, where 14 represents Ace
    suits = [0, 1, 2, 3]  # spades, diamonds, hearts, clubs


    # Generate all possible cards
    all_cards = [Card(value, suit) for value in values for suit in suits]

    # Generate all unique combinations of two cards
    combinations = [(card1, card2) for i, card1 in enumerate(all_cards)
                    for card2 in all_cards[i+1:] if card1.value != card2.value]

    # Sort non-pair combinations so that the stronger card is shown first
    for i in range(len(combinations)):
        if combinations[i][0].value < combinations[i][1].value:
            combinations[i] = (combinations[i][1], combinations[i][0])

    # Generate all pairs
    pairs = [(Card(value, suit1), Card(value, suit2))
             for value in values for suit1 in suits for suit2 in suits
             if suit1 != suit2]

    # Sort pairs by strength
    pairs.sort(key=lambda pair: -pair[0].value)

    # Sort non-pair combinations by strength
    combinations.sort(key=lambda pair: (-max(pair[0].value, pair[1].value),
                                         -min(pair[0].value, pair[1].value)))

    # Combine pairs and non-pairs, with pairs first
    all_combinations = pairs + combinations

    return all_combinations

holecards = generate_card_combinations()
stacks = [20, 20]
game = Game(n_players=2, stacks=stacks, hand_history=True)
game.new_hand(first_hand=True)
agent_0 = Agent()
agent_1 = Agent()
map_agent = {game.players[0]: agent_0, game.players[1]: agent_1}

hand_0 = game.players[0].holecards
hand_1 = game.players[1].holecards
map_cards = {agent_0: hand_0, agent_1: hand_1}

# to store the current policies in a well readable form
# SB and BB
states = [[], []]
representations = []
for hand in holecards:
    cards = agent_0._one_hot_encode_hand(*hand)
    for i in range(2):
        state = np.append(cards, i).reshape(1, -1)
        states[i].append(state)
    representations.append((hand[0].representation, hand[1].representation))
dfs = []
for i in range(2):
    dfs.append(pd.DataFrame({"hand": representations}))
    states[i] = np.squeeze(np.array(states[i]))


for k in tqdm.tqdm(range(200000)):
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
        #legal_actions = game.get_legal_actions()
        minbet = game.get_legal_betsize()
        action, bet_size, bet_frac = agent_to_act.choose_action(cards, position, legal_actions, minbet, next.stack, next.bet)
        if action == 2:
            game.implement_action(next, action, 20)
        else:
            game.implement_action(next, action)

        agent_to_act.add_to_episode((cards, position), action, bet_frac)


    for player, agent in map_agent.items():
        position = game.positions.index(player)
        if len(agent.episode["actions"]) > 0:
            reward = player.stack -player.starting_stack
            reward = 0.1*(reward + 1 if game.positions.index(player) == 0 else reward + 2)
            agent.train_on_episode(reward)
    game.new_hand()

    if k%1000 ==0:
        for a in range(2):
            agent = agent_0 if a == 0 else agent_1
            agent.save_model(name_actor="actor_" + str(a), name_critic="critic_" + str(a))
            for i in range(2):
                decision, alpha, beta = agent.actor(states[i])
                ev = agent.critic(states[i])
                df = dfs[i]
                df[["fold_" + str(a), "call_check_" + str(a), "bet_raise_ " + str(a)]] = decision.numpy()
                df["EV_" + str(a)] = ev
                df["alpha_" + str(a)] = alpha.numpy()
                df["beta_" + str(a)] = beta.numpy()
                df["raise_frac_" + str(a)] = np.random.beta(alpha, beta)


        dfs[0].to_csv("actions_preflop_sb_" + str(k) + ".csv", index=False)
        dfs[1].to_csv("actions_preflop_bb_" + str(k) + ".csv", index=False)




