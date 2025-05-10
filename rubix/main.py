# rubix/main.py

"""
Authors' Note:

Despite the data provided for the resolution of the problem turning out to 
have all players distributed into all teams, we assumed that our program should be able to 
work in the general case—as was the case before the teacher corrected the data :) —
and that the shape of the largest solution possible (i.e., the largest number of teams 
for which a solution exists) was one less, due to the fact that one player would be
left out.

For this reason, the dimensions of the problem space are not defined beforehand but are inferred
from the dataset provided and the constraints imposed by the caller. This inference is done dynamically
based on the number of players and feasible team configurations.

Moreover, we believe this solution will differ significantly from others. Despite using
OOP, we have abstracted the league as a window within a matrix—called the fitness window—
which's size is determined either through programmatic inference or through user input. This fitness
window defines the operational subspace in which valid team configurations are evaluated.

"""

import argparse
from typing import Any
from rubix.constants import DATA_V1
from rubix.loader import load_data
from rubix.processor import process_data
from rubix.classes.solver import Solver


def get_args(
        
) -> argparse.Namespace:
    
    """
    Parses command-line arguments for the optimization program.

    Returns:
        argparse.Namespace: Parsed arguments, including:
            test (str): A test argument for demonstration or debugging.
    """ 

    parser = argparse.ArgumentParser(description="Run the optimization solver")
    parser.add_argument(
        "--test", 
        type=str, 
        default="Hello World!", 
        help="Test args"
    )
    return parser.parse_args()


def main(
    **kwargs: dict[str, Any]
) -> None:
    
    """
    Executes the optimization routine.

    This function prepares the input data, initializes the solver, and executes
    the optimization process. It accepts arbitrary keyword arguments, which may
    include optional parameters for the solver or runtime configuration.

    Expected kwargs:
        path (str): Optional path to the input dataset. Defaults to DATA_V1.

    Returns:
        None
    """

    # Load the data from the provided path
    dataset = load_data(
        path=DATA_V1, 
        config_path="rubix.configs/baseline_config.json"
    )

    # Process the data to the necessary shape
    dataset = process_data(dataset)
    
    # Initialize the solver
    solver = Solver(
        dataset.matrix, 
        dataset.cost_params,  
        dataset.layout_params, 
        dataset.solver_params
    )

    # Solve the problem
    result = solver.solve(**kwargs)

    # Print final solution
    print(dataset, result)

if __name__ == "__main__":
    main()