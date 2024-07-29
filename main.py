import argparse

from Algorithms import FlatMonteCarlo, RAVE, UCT, UCB


def parse_args():
    parser = argparse.ArgumentParser(description="Abalone Solver Monte Carlo Tree Search")
    parser.add_argument("--solver",
                        type=str,
                        choices=["FLAT_MC", "RAVE", "UCT", "UCB"],
                        default="UCB",
                        help="Solver algorithm to use")
    parser.add_argument("--render_mode", type=str, default="human", help="Render mode of the game")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum number of iterations for the solver")
    parser.add_argument("--random_player", action='store_true', help="Randomly pick player to start the game")
    parser.add_argument("--random_pick", action='store_true', help="Randomly pick game layout")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    solver_mapping = {
        "FLAT_MC" : FlatMonteCarlo,
        "RAVE" : RAVE,
        "UCT" : UCT,
        "UCB" : UCB
    }
    selected_solver = solver_mapping[args.solver]
    selected_solver.main(args.render_mode, args.max_iterations, args.random_player, args.random_pick)
