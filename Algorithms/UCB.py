import copy
import gym
import random
from gym_abalone.envs.abalone_env import AbaloneEnv
import numpy as np

class Node:
    def __init__(self, state, move=None):
        self.state = state  # State of the game
        self.move = move  # Move that led to this state
        self.visit_count = 0  # Number of visits
        self.total_reward = 0  # Cumulative reward
        self.children = []  # Child nodes

    def __repr__(self):
        return f"Node(move={self.move}, visit_count={self.visit_count}, total_reward={self.total_reward})"

class UCBAgent:
    def __init__(self, env, max_iterations=30, exploration_param=1.4):
        self.env = env
        self.max_iterations = max_iterations
        self.exploration_param = exploration_param

    def ucb_search(self):
        root_node = Node(clone_game(self.env.game))  # Clone the initial game state
        print(f"\troot_node: {root_node}")
        possible_moves = self.env.game.get_possible_moves(root_node.state.current_player)
        possible_moves = self.env.game.get_possible_moves(root_node.state.current_player)
        nodes = {move: Node(clone_game(root_node.state), move) for move in possible_moves}

        for _ in range(self.max_iterations):
            print(f"\t\titeration {_}")
            selected_node = self.selection(nodes)
            print(f"\t\t\t selected node: {selected_node}")
            reward = self.simulation(selected_node)
            print(f"\t\t\t reward: {reward}")
            self.backpropagation(selected_node, reward)

        best_move = max(nodes.keys(), key=lambda move: nodes[move].total_reward / nodes[move].visit_count if nodes[move].visit_count > 0 else 0)
        print(f"\tbest_move: {best_move}")
        return best_move

    def ucb(self, node, total_visits):
        if node.visit_count == 0:
            return float('inf')
        average_reward = node.total_reward / node.visit_count
        exploration_term = self.exploration_param * np.sqrt(np.log(total_visits) / node.visit_count)
        return average_reward + exploration_term

    def selection(self, nodes):
        total_visits = sum(node.visit_count for node in nodes.values()) + 1
        ucb_values = {move: self.ucb(node, total_visits) for move, node in nodes.items()}
        selected_move = max(ucb_values.keys(), key=lambda move: ucb_values[move])
        return nodes[selected_move]

    def simulation(self, node):
        sim_state = clone_game(node.game)
        sim_state.action_handler(node.move[0], node.move[1])
        while not sim_state.game_over:
            possible_moves = sim_state.get_possible_moves(sim_state.current_player)
            if not possible_moves:
                break
            random_move = random.choice(possible_moves)
            sim_state.action_handler(random_move[0], random_move[1])

        winner = np.argmax(sim_state.players_victories)
        return 1.0 if winner == node.game.current_player else 0.0

    def backpropagation(self, node, reward):
        current_node = node
        while current_node:
            current_node.visit_count += 1
            current_node.total_reward += reward
            current_node = None  # No parent relationship in this UCB implementation

def clone_game(game):
    return copy.deepcopy(game)

def str_game(game):
    return f"Game({game.current_player}, {game.players_victories})"


env = AbaloneEnv(render_mode='human')
# env = AbaloneEnv(render_mode='terminal')
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
