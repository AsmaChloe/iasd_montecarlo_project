import copy
import random
from gym_abalone.envs.abalone_env import AbaloneEnv
import numpy as np
from gym_abalone.game.engine.gamelogic import AbaloneGame
import time

class Node:
    def __init__(self, env, move=None, parent=None, move_type=None):
        self.original_env = clone_env(env)
        self.env = env
        self.move = move
        self.move_type = move_type
        self.visit_count = 0
        self.total_reward = 0
        self.children = []
        self.parent = parent
        self.amaf_visit_count = 0
        self.amaf_total_reward = 0

    def __repr__(self):
        return f"Node(id={hex(id(self))} env={self.env}, move={self.move}, visit_count={self.visit_count}, total_reward={self.total_reward}, move_type={self.move_type})"

    def reset_env(self):
        self.env = clone_env(self.original_env)

class UCTAgent:
    def __init__(self, env, max_iterations=30):
        self.env = env
        self.max_iterations = max_iterations

    def uct_search(self, root_node):
        for _ in range(self.max_iterations):
            # print(f"iteration {_}")

            root_node.reset_env()
            selected_node = self.selection(root_node)
            # print(f"\tselected_node: {selected_node}")
            new_node = self.expansion(selected_node)
            reward, visited_nodes = self.simulation(new_node)
            self.backpropagation(new_node, reward, visited_nodes)

        # best_child = max(root_node.children, key=lambda child: child.total_reward / child.visit_count if child.visit_count > 0 else 0)
        # return best_child
        ## Choose the action with the highest average reward

        # Select candidate children with the highest average reward
        max_criteria = max(root_node.children,
                           key=lambda child: child.total_reward / child.visit_count if child.visit_count > 0 else 0)
        candidate_children_by_move_type = {"winner": [], "ejected": [], "inline_push": [], "sidestep_move": [],
                                           "inline_move": []}
        for child in root_node.children:
            avg_reward = child.total_reward / child.visit_count if child.visit_count > 0 else 0
            if avg_reward == max_criteria.total_reward / max_criteria.visit_count:
                candidate_children_by_move_type[child.move_type].append(child)

        # Choose the best child among the candidate children
        for move_type, candidate_children in candidate_children_by_move_type.items():
            if len(candidate_children) > 0:
                best_child = random.choice(candidate_children)
                break

        # print(f"\tbest_child: {best_child} - {best_child.env.game.current_player}")
        return best_child

    def ucb_rave(self, node, constant=1.4, rave_constant=1.0):
        if node.visit_count == 0:
            return float('inf')
        uct_value = node.total_reward / node.visit_count + constant * np.sqrt(np.log(node.parent.visit_count) / node.visit_count)
        if node.amaf_visit_count > 0:
            rave_value = node.amaf_total_reward / node.amaf_visit_count
            beta = node.amaf_visit_count / (node.visit_count + node.amaf_visit_count + 4 * node.visit_count * node.amaf_visit_count * rave_constant)
            return beta * rave_value + (1 - beta) * uct_value
        return uct_value

    def selection(self, node):
        current_node = node
        while current_node.children:
            current_node = max(current_node.children, key=lambda n: self.ucb_rave(n))
        return current_node

    def expansion(self, node):
        if node.visit_count == 0:
            return node
        possible_moves = node.env.game.get_possible_moves(node.env.game.current_player, group_by_type=True)
        for move_type in possible_moves:
            for move in possible_moves[move_type]:
                new_env = clone_env(node.env)
                new_env.step(move)
                new_node = Node(new_env, move, node, move_type=move_type)
                node.children.append(new_node)
        chosen_node = random.choice(node.children)
        return chosen_node

    def simulation(self, node):
        visited_nodes = set()
        while not node.env.game.game_over:
            possible_moves = node.env.game.get_possible_moves(node.env.game.current_player, group_by_type=True)
            if len(possible_moves) == 0:
                break
            for move_type in ['winner', 'ejected', 'inline_push', 'sidestep_move', 'inline_move']:
                if possible_moves[move_type]:
                    move = random.choice(possible_moves[move_type])
                    node.env.step(move)
                    visited_nodes.add((move_type, move))
                    break
        winner = np.argmax(node.env.game.players_victories)
        return 1.0 if winner == node.original_env.game.current_player else 0.0, visited_nodes

    def backpropagation(self, node, reward, visited_nodes):
        current_node = node
        while current_node:
            current_node.visit_count += 1
            current_node.total_reward += reward
            for move_type, move in visited_nodes:
                ancestor = current_node
                while ancestor:
                    if ancestor.move == move:
                        ancestor.amaf_visit_count += 1
                        ancestor.amaf_total_reward += reward
                    ancestor = ancestor.parent
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
    root_node = Node(clone_env(env), move=move, parent=parent)
    while not done:
        best_child = uct_agent.uct_search(root_node)
        obs, reward, done, info = env.step(best_child.move)
        parent = root_node
        move = best_child.move
        root_node = best_child
        print(f"{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} | reward={reward: >4} ")
        env.render(fps=0.5)

    print(f"Episode {info['turn']: <4} finished after {env.game.turns_count} turns \n")
env.close()
