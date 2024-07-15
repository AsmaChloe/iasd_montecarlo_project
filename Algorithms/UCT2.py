"""
demo of the env
"""
import copy

import random
from gym_abalone.envs.abalone_env import AbaloneEnv
import numpy as np

from gym_abalone.game.engine.gamelogic import AbaloneGame


class Node:
    def __init__(self, env, move=None, parent=None):
        self.original_env = clone_env(env)
        self.env = env
        # move that led to this state
        self.move = move
        # number of visits
        self.visit_count = 0
        # cumulative reward
        self.total_reward = 0
        self.children = []
        self.parent = parent

    def __repr__(self):
        return f"Node(id={hex(id(self))} env={self.env}, move={self.move}, visit_count={self.visit_count}, total_reward={self.total_reward})"

    def reset_env(self):
        self.env = clone_env(self.original_env)

class UCTAgent:
    def __init__(self, env, max_iterations=30):
        self.env = env
        self.max_iterations = max_iterations

    def uct_search(self, root_node):
        for _ in range(self.max_iterations):
            root_node.reset_env()
            selected_node = self.selection(root_node)
            new_node = self.expansion(selected_node)
            reward = self.simulation(new_node)
            self.backpropagation(new_node, reward)

        # Choose the action with the highest average reward
        best_child = max(root_node.children, key=lambda child: child.total_reward / child.visit_count if child.visit_count > 0 else 0)
        # print(f"\tbest_child: {best_child} - {best_child.env.game.current_player}")
        return best_child

    def ucb(self, node, constant = 1.4):
        # Implement the Upper Confidence Bound for Trees (UCT) formula
        if node.visit_count == 0:
            return float('inf')
        ucb = node.total_reward / node.visit_count + constant * np.sqrt(np.log(node.parent.visit_count) / node.visit_count)
        return ucb

    def selection(self, node):
        current_node = node
        # While not a leaf node, choose the child node with the highest UCB value
        while current_node.children:
            current_node = max(current_node.children, key=self.ucb)
        return current_node

    def expansion(self, node):
        # If never been visited, return itself
        if node.visit_count == 0:
            return node

        # Else, expand the node by adding all possible child nodes
        possible_moves = node.env.game.get_possible_moves(node.env.game.current_player)
        for move in possible_moves:
            new_env = clone_env(node.env)
            new_env.step(move)
            new_node = Node(new_env, move, node)
            node.children.append(new_node)

        chosen_node = random.choice(node.children)
        return chosen_node

    def simulation(self, node):
        while not node.env.game.game_over:
            possible_moves = node.env.game.get_possible_moves(node.env.game.current_player)
            # No possible moves, end the game
            if len(possible_moves) == 0:
                break
            move = random.choice(possible_moves)
            node.env.step(move)
            if not node.env.game.game_over and node.env.done:
                return 0.0

        winner = np.argmax(node.env.game.players_victories)
        # print(f"\t\t\t winner: {winner}")
        return 1.0 if winner == node.env.game.current_player else 0.0

    def backpropagation(self, node, reward):
        # Update visit count and total reward in all nodes in path from root to node
        current_node = node
        # print(f"\t\t backpropagation: {current_node} - reward: {reward}")
        while current_node:
            current_node.visit_count += 1
            current_node.total_reward += reward
            # print(f"\t\t\t node updated : {current_node}")
            current_node = current_node.parent

def clone_game(game):
    cloned_game = AbaloneGame()
    cloned_game.board = copy.deepcopy(game.board)
    cloned_game.positions = copy.deepcopy(game.positions)

    cloned_game.variant = game.variant
    cloned_game.players = game.players
    cloned_game.players_sets = copy.deepcopy(game.players_sets)

    cloned_game.players_damages = copy.deepcopy(game.players_damages)

    cloned_game.turns_count = game.turns_count
    cloned_game.current_player = game.current_player
    cloned_game.game_over = game.game_over

    cloned_game.episode = game.episode
    cloned_game.players_victories = copy.deepcopy(game.players_victories)

    return cloned_game

def clone_env(env):
    cloned_env = AbaloneEnv(render_mode=env.render_mode, max_turns=env.max_turns)
    cloned_env.action_space = env.action_space
    cloned_env.observation_space = env.observation_space
    cloned_env.game = clone_game(env.game)
    cloned_env.gui = env.gui

    return cloned_env

# To see game interface
env = AbaloneEnv(render_mode='human', max_turns=2000)
# env = AbaloneEnv(render_mode='terminal', max_turns=2000)
uct_agent = UCTAgent(env, max_iterations=3)

NB_EPISODES = 1
for episode in range(1, NB_EPISODES + 1):
    env.reset(random_player=True, random_pick=False)
    done = False
    parent = None
    move = None
    root_node = Node(clone_env(env), move = move, parent=parent)
    while not done:
        # print(f"{root_node=}")
        best_child = uct_agent.uct_search(root_node)
        obs, reward, done, info = env.step(best_child.move)
        parent = root_node
        move = best_child.move
        root_node = best_child
        print(f"{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} | reward={reward: >4} ")
        env.render(fps=0.5)

    print(f"Episode {info['turn']: <4} finished after {env.game.turns_count} turns \n")
env.close()

