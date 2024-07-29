import copy
import random
from gym_abalone.envs.abalone_env import AbaloneEnv
import numpy as np
class UCBAgent:
    def __init__(self, env, max_iterations=30, exploration_param=1.4):
        self.env = env
        self.max_iterations = max_iterations
        self.exploration_param = exploration_param

    def ucb_search(self):

        # Get all possible moves as children
        possible_moves = self.env.game.get_possible_moves(self.env.game.current_player, group_by_type=True)
        children = { move : {'game' : clone_game(self.env.game),'visit_count' : 0, 'total_reward' : 0, 'move_type' : move_type} for move_type, moves in possible_moves.items() for move in moves}

        for _ in range(self.max_iterations):
            # print(f"iteration {_}")

            # Selection
            selected_move = self.selection(children)

            # Simulation
            reward = self.simulation(selected_move, children[selected_move])

            # Updating stats
            self.update_stats(children[selected_move], reward)

        ## Choose the action with the highest average reward

        # Select candidate children with the highest average reward
        # Move with the highest average reward
        criterion_move = max(children.keys(), key=lambda move: children[move]["total_reward"] / children[move]["visit_count"] if children[move]["visit_count"] > 0 else 0)
        moves_by_move_type = {"winner": [], "ejected": [], "inline_push": [], "sidestep_move": [], "inline_move": []}

        # Get all moves with the same average reward as the criterion move
        for move, move_stats in children.items():
            avg_reward = children[move]["total_reward"] / children[move]["visit_count"] if children[move]["visit_count"] > 0 else 0
            if avg_reward == children[criterion_move]["total_reward"] / children[criterion_move]["visit_count"]:
                moves_by_move_type[children[move]["move_type"]].append(move)

        # Choose the best child among the candidate moves
        for move_type, moves in moves_by_move_type.items():
            if len(moves) > 0:
                best_move = random.choice(moves)
                break

        return best_move

    def ucb(self, node, total_visits):
        if node["visit_count"] == 0:
            return float('inf')
        average_reward = node["total_reward"] / node["visit_count"]
        exploration_term = self.exploration_param * np.sqrt(np.log(total_visits) / node["visit_count"])
        return average_reward + exploration_term

    def selection(self, children):
        # As I am computing the UCB value for each child, I update visit_count
        total_visits = sum(node["visit_count"] for node in children.values()) + 1

        # Choose the child with the highest UCB value
        ucb_values = {move: self.ucb(node, total_visits) for move, node in children.items()}
        selected_move = max(ucb_values.keys(), key=lambda move: ucb_values[move]) # TODO selectionner en fonction du type
        return selected_move

    def simulation(self, selected_move, child):
        sim_state = clone_game(child["game"])
        sim_state.action_handler(selected_move[0], selected_move[1])

        while not sim_state.game_over:
            possible_moves = sim_state.get_possible_moves(sim_state.current_player, group_by_type=True)
            # No possible moves, end the game
            if len(possible_moves) == 0:
                break
            for move_type in ['winner', 'ejected', 'inline_push', 'sidestep_move', 'inline_move']:
                if possible_moves[move_type]:
                    random_move = random.choice(possible_moves[move_type])
                    sim_state.action_handler(random_move[0], random_move[1])
                    break

        winner = np.argmax(sim_state.players_victories)
        return 1.0 if winner == child["game"].current_player else 0.0

    def update_stats(self, child, reward):
        child["visit_count"] += 1
        child["total_reward"] += reward

def clone_game(game):
    return copy.deepcopy(game)

def str_game(game):
    return f"Game({game.current_player}, {game.players_victories})"

def main(render_mode='human', max_iterations=3, random_player=True, random_pick=True):
    env = AbaloneEnv(render_mode=render_mode, max_turns=2000)
    ucb_agent = UCBAgent(env, max_iterations=max_iterations)

    NB_EPISODES = 1
    for episode in range(1, NB_EPISODES + 1):
        env.reset(random_player=random_player, random_pick=random_pick)
        done = False
        while not done:
            action = ucb_agent.ucb_search()
            obs, reward, done, info = env.step(action)
            print(f"{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} | reward={reward: >4} ")
            env.render(fps=1)
        print(f"Episode {info['turn']: <4} finished after {env.game.turns_count} turns \n")
    env.close()

if __name__ == "__main__":
    main(render_mode='human', max_iterations=3)