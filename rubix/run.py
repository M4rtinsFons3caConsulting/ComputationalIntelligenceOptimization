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

from typing import Any
import torch

from rubix.loader import load_data
from rubix.processor import process_data
from rubix.classes.solver import Solver

def run(
    **kwargs: dict[str, Any]
) -> None:
    
    """
    Executes the optimization routine.

    This function prepares the input data, initializes the solver, and executes
    the optimization process. It accepts arbitrary keyword arguments, which may
    include optional parameters for the solver or runtime configuration.

    Expected kwargs:
        data_path (str): Path to dataset
        config_path (str): Path to configuration file.

    Returns:
        DataSet, Result
    """

    data_path = kwargs.pop('data_path', None)
    config_path = f"rubix.configs/{kwargs.pop('path',  'hill_climber_config.json')}"

    # Load the data from the provided path
    dataset = load_data(
        path=data_path,
        config_path = config_path
    )

    # Process the data to the necessary shape
    dataset = process_data(dataset)
    
    # Print the resulting dataset object.
    print(dataset)
    
    # Initialize the solver
    solver = Solver(
        dataset.matrix, 
        dataset.cost_params,  
        dataset.layout_params, 
        dataset.solver_params
    )

    # Solve the problem
    result, history = solver.solve(**kwargs)

    # Parse history to list
    history = torch.stack(history).numpy().tolist()

    return dataset, result, history