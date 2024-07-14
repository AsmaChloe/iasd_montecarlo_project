import copy
import gym
import random
from gym_abalone.envs.abalone_env import AbaloneEnv
import numpy as np

class FlatMonteCarloAgent:
    def __init__(self, env, max_simulations=100):
        self.env = env
        self.max_simulations = max_simulations

    def flat_monte_carlo_search(self):
        root_state = clone_game(self.env.game)
        print(f"root_state: {str_game(root_state)}")
        possible_moves = self.env.game.get_possible_moves(root_state.current_player)
        move_rewards = {move: [] for move in possible_moves}

        for move in possible_moves:
            print(f"move: {move}")
            for _ in range(self.max_simulations):
                print(f"\titeration {_}")
                reward = self.simulation(root_state, move)
                print(f"\treward: {reward}")
                move_rewards[move].append(reward)

        avg_rewards = {move: np.mean(rewards) for move, rewards in move_rewards.items()}
        best_move = max(avg_rewards.keys(), key=lambda move: avg_rewards[move])
        return best_move

    def simulation(self, state, move):
        sim_state = clone_game(state)
        sim_state.action_handler(move[0], move[1])
        while not sim_state.game_over:
            possible_moves = sim_state.get_possible_moves(sim_state.current_player)
            if not possible_moves:
                break
            random_move = random.choice(possible_moves)
            sim_state.action_handler(random_move[0], random_move[1])

        winner = np.argmax(sim_state.players_victories)
        return 1.0 if winner == state.current_player else 0.0

def clone_game(game):
    return copy.deepcopy(game)

def str_game(game):
    return f"Game({game.current_player}, {game.players_victories})"

env = AbaloneEnv(render_mode='human')
# env = AbaloneEnv(render_mode='terminal')
flat_monte_carlo_agent = FlatMonteCarloAgent(env, max_simulations=10)

NB_EPISODES = 1
for episode in range(1, NB_EPISODES + 1):
    env.reset(random_player=True, random_pick=True)
    done = False
    while not done:
        action = flat_monte_carlo_agent.flat_monte_carlo_search()
        obs, reward, done, info = env.step(action)
        print(f"{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} | reward={reward: >4} ")
        env.render(fps=1)
    print(f"Episode {info['turn']: <4} finished after {env.game.turns_count} turns \n")
env.close()
