import copy
import gym
import random
from gym_abalone.envs.abalone_env import AbaloneEnv
import numpy as np

from gym_abalone.game.engine.gamelogic import AbaloneGame

class Node:
    def __init__(self, env, move=None, move_type=None):
        self.original_env = clone_env(env)
        self.env = clone_env(env)

        self.move = move
        self.move_type = move_type

        #Reward for each simulation
        self.rewards = []
        # number of visits
        # self.visit_count = 0
        # cumulative reward
        # self.total_reward = 0
        # self.children = []
        # self.parent = parent

    def __repr__(self):
        return f"Node(id={hex(id(self))} env={self.env}, move={self.move}, move_type={self.move_type}), rewards={self.rewards})"

    def reset_env(self):
        self.env = clone_env(self.original_env)

class FlatMonteCarloAgent:
    def __init__(self, env, max_simulations=100):
        self.env = env
        self.max_simulations = max_simulations
    def flat_monte_carlo_search(self):
        move_nodes = []
        # Get all possible moves
        possible_moves = self.env.game.get_possible_moves(self.env.game.current_player, group_by_type=True)

        # Play each move max_simulations times
        for move_type, moves in possible_moves.items():
            for move in moves :
                # move_rewards[(move_type, move)] = []

                new_node = Node(self.env, move=move, move_type=move_type)

                for _ in range(self.max_simulations):
                    # print(f"\titeration {_}")

                    # Simulate the game
                    reward = self.simulation(new_node)
                    # print(f"\treward: {reward}")
                    new_node.rewards.append(reward)

                move_nodes.append(new_node)

        avg_rewards = {node: np.mean(node.rewards) for node in move_nodes}
        max_criteria = max(avg_rewards.values())
        candidates_nodes = {"winner" : [], "ejected" : [], "inline_push" : [], "sidestep_move" : [], "inline_move" : []}
        for node, avg_reward in avg_rewards.items():
            if avg_reward == max_criteria:
                candidates_nodes[node.move_type].append(node)

        for move_type, nodes in candidates_nodes.items():
            if len(nodes) > 0:
                best_node = random.choice(nodes)
                break

        return best_node

    def simulation(self, node):

        node.env.step(node.move)
        while not node.env.game.game_over:
            possible_moves = node.env.game.get_possible_moves(node.env.game.current_player, group_by_type=True)
            # No possible moves, end the game
            if len(possible_moves) == 0:
                break
            for move_type in ['winner', 'ejected', 'inline_push', 'sidestep_move', 'inline_move']:
                if possible_moves[move_type]:
                    move = random.choice(possible_moves[move_type])
                    node.env.step(move)
                    break

        winner = np.argmax(node.env.game.players_victories)
        # print(f"\t\t\t winner: {winner}")
        return 1.0 if winner == self.env.game.current_player else 0.0

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

env = AbaloneEnv(render_mode='human', max_turns=2000)
# env = AbaloneEnv(render_mode='terminal', max_turns=2000)
flat_monte_carlo_agent = FlatMonteCarloAgent(env, max_simulations=3)

NB_EPISODES = 1
for episode in range(1, NB_EPISODES + 1):
    env.reset(random_player=False, random_pick=False)
    done = False
    while not done:
        best_node = flat_monte_carlo_agent.flat_monte_carlo_search()
        obs, reward, done, info = env.step(best_node.move, verbose=True)
        print(f"{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} | reward={reward: >4} ")
        env.render(fps=1)
    print(f"Episode {info['turn']: <4} finished after {env.game.turns_count} turns \n")
env.close()
