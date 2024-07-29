# Abalone Solver Monte Carlo Tree Search

This project is a Monte Carlo Tree Search solver for the game Abalone, developed as part of the Monte Carlo Search course taught by Tristan Cazenave at Université Paris-Dauphine, PSL Master 2 IASD. The project implements various Monte Carlo search algorithms to solve problems in the game of Abalone, providing a command-line interface to configure simulation parameters.

## Prerequisites and Running Experiments

To run the Monte Carlo search simulations described in this project, you need to set up your development environment and install the necessary dependencies. Follow these steps:

### Prerequisites

1. **Python**: This project has been developed with Python 3.12. Ensure that Python 3.12 is installed on your system.
   
   Check your Python version with:
   ```bash
   python --version
    ```

2. **Python Libraries**: You need to install the required Python libraries to run the project. 
    ```bash
    pip install -r requirements.txt
    ```

### Running an Experiment

The project uses a command-line interface to configure the simulation parameters. Here are the available options:

- `--solver`: Solver algorithm to use, with the following choices:
  - `FLAT_MC`: Flat Monte Carlo
  - `RAVE`: Rapid Action Value Estimation
  - `UCT`: Upper Confidence bounds applied to Trees (default)
  - `UCB`: Upper Confidence Bound

- `--render_mode`: Render mode of the game (`human` by default), specifies how the game is displayed.

- `--max_iterations`: Maximum number of iterations for the solver (default: 3).

- `--random_player`: Randomly pick the player to start the game (`True` if specified).

- `--random_pick`: Randomly pick the game layout (`True` if specified).

For example, to run a simulation using the RAVE algorithm, with default rendering, a maximum of 5 iterations, randomly picking the starting player, and a fixed game layout, you can execute the following command:

```bash
python main.py --solver RAVE --max_iterations 5 --random_player
```
