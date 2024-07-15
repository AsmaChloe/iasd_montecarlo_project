import copy
import gym
import random
from gym_abalone.envs.abalone_env import AbaloneEnv
import numpy as np

class UCBAgent:
    def __init__(self, env, max_iterations=30, exploration_param=1.4):
        self.env = env
        self.max_iterations = max_iterations
        self.exploration_param = exploration_param

    def ucb_search(self):
        current_env = self.env
        # print(f"\tcurrent_env: {current_env}")
        possible_moves = self.env.game.get_possible_moves(current_env.game.current_player)
        #Children and their stats
        children = {move: {"game":clone_game(current_env.game), "visit_count" : 0, "total_reward" : 0} for move in possible_moves}

        for _ in range(self.max_iterations):
            # print(f"\t\titeration {_}")
            # Select a child move based on UCB
            selected_move = self.selection(children)
            # print(f"\t\t\t selected node: {selected_move}")
            # Simulate the game from the selected move
            reward = self.simulation(selected_move, children[selected_move])
            # print(f"\t\t\t reward: {reward}")
            # Update the stats of the selected move
            self.update_stats(children[selected_move], reward)

        # Choose the action with the highest average reward
        best_move = max(children.keys(), key=lambda move: children[move]["total_reward"] / children[move]["visit_count"] if children[move]["visit_count"] > 0 else 0)
        # print(f"\tbest_move: {best_move}")
        return best_move

    def ucb(self, node, total_visits):
        if node["visit_count"] == 0:
            return float('inf')
        average_reward = node["total_reward"] / node["visit_count"]
        exploration_term = self.exploration_param * np.sqrt(np.log(total_visits) / node["visit_count"])
        return average_reward + exploration_term

    def selection(self, children):
        # Get the total number of visits of all children
        total_visits = sum(node["visit_count"] for node in children.values()) + 1

        # Choose the child with the highest UCB value
        ucb_values = {move: self.ucb(node, total_visits) for move, node in children.items()}
        selected_move = max(ucb_values.keys(), key=lambda move: ucb_values[move])
        return selected_move

    def simulation(self, selected_move, child):
        sim_state = clone_game(child["game"])
        sim_state.action_handler(selected_move[0], selected_move[1])
        while not sim_state.game_over:
            possible_moves = sim_state.get_possible_moves(sim_state.current_player)
            if not possible_moves:
                break
            random_move = random.choice(possible_moves)
            sim_state.action_handler(random_move[0], random_move[1])

        winner = np.argmax(sim_state.players_victories)
        return 1.0 if winner == child["game"].current_player else 0.0

    def update_stats(self, child, reward):
        child["visit_count"] += 1
        child["total_reward"] += reward

def clone_game(game):
    return copy.deepcopy(game)

def str_game(game):
    return f"Game({game.current_player}, {game.players_victories})"


# env = AbaloneEnv(render_mode='human')
env = AbaloneEnv(render_mode='terminal')
ucb_agent = UCBAgent(env, max_iterations=3)

NB_EPISODES = 1
for episode in range(1, NB_EPISODES + 1):
    env.reset(random_player=True, random_pick=True)
    done = False
    while not done:
        action = ucb_agent.ucb_search()
        obs, reward, done, info = env.step(action)
        print(f"{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} | reward={reward: >4} ")
        env.render(fps=1)
    print(f"Episode {info['turn']: <4} finished after {env.game.turns_count} turns \n")
env.close()
