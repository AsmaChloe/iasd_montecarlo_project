"""
demo of the env
"""
import gym
import random
from gym_abalone.envs.abalone_env import AbaloneEnv
import numpy as np

class Node:
    def __init__(self, state, move=None):
        # state of the game
        self.state = state
        # list of possible moves
        self.move = move
        # number of visits
        self.visit_count = 0
        # cumulative reward
        self.total_reward = 0
        self.children = []
        self.parent = None

    def __repr__(self):
        return f"Node({self.state}, move={self.move}, visit_count={self.visit_count}, total_reward={self.total_reward})"

class UCTAgent:
    def __init__(self, env, max_iterations=100):
        self.env = env
        self.max_iterations = max_iterations

    def uct_search(self):
        root_node = Node(self.env.game.clone())  # Clone the initial game state
        for _ in range(self.max_iterations):
            print(f"iteration {_}")
            selected_node = self.selection(root_node)
            new_node = self.expansion(selected_node)
            reward = self.simulation(new_node)
            self.backpropagation(new_node, reward)

        # Choose the action with the highest average reward
        best_child = max(root_node.children, key=lambda child: child.total_reward / child.visit_count)
        return best_child.move

    def ucb(self, node):
        # Implement the Upper Confidence Bound for Trees (UCT) formula
        constant = 1
        if node.visit_count == 0:
            return float('inf')
        ucb = node.total_reward / node.visit_count + constant * np.sqrt(np.log(node.parent.visit_count) / node.visit_count)
        return ucb

    def selection(self, node):
        print(f"selection node: {node}")
        # Implement UCB1 selection logic
        # Traverse tree based on UCB1 formula
        current_node = node
        while current_node.children:
            current_node = max(current_node.children, key=self.ucb)
        print(f"\t selected node: {current_node}")
        return current_node

    def expansion(self, node):
        print(f"expansion node: {node}")
        # Expand node by adding child nodes for all possible moves
        possible_moves = self.env.game.get_possible_moves(node.state.current_player)
        print(f"\t possible_moves: {possible_moves}")
        for move in possible_moves:
            new_state = node.state.clone()
            new_state.action_handler(move[0], move[1])
            new_node = Node(new_state, move)
            node.children.append(new_node)
            new_node.parent = node
        return random.choice(node.children) if node.children else node  # Choose a random child initially

    def simulation(self, node):
        # Simulate game until terminal state using random moves
        # Return reward

        state = node.state.clone()
        while not state.game_over:
            possible_moves = state.get_possible_moves(state.current_player)
            move = random.choice(possible_moves)
            state.action_handler(move[0], move[1])

        # Calculate reward (for simplicity, assuming reward based on game outcome)
        winner = np.argmax(state.players_victories)
        if winner == node.state.current_player:
            return 1.0  # Win
        else:
            return 0.0  # Loss or draw

    def backpropagation(self, node, reward):
        # Update visit count and total reward in all nodes in path from root to node
        current_node = node
        while current_node:
            current_node.visit_count += 1
            current_node.total_reward += reward
            current_node = current_node.parent


# env = gym.make('abalone-v0')
env = AbaloneEnv(render_mode='terminal')
uct_agent = UCTAgent(env)

print(env.action_space)
# > Discrete(2)
print(env.observation_space)
# > Box(11,11)


NB_EPISODES = 1
for episode in range(1, NB_EPISODES + 1):
    env.reset(random_player=True, random_pick=True)
    done = False
    while not done:
        action = uct_agent.uct_search()
        obs, reward, done, info = env.step(action)
        print(f"{info['turn']: <4} | {info['player_name']} | {str(info['move_type']): >16} | reward={reward: >4} ")
        env.render(fps=1)
    print(f"Episode {info['turn']: <4} finished after {env.game.turns_count} turns \n")
env.close()

