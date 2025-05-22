from typing import Any, Dict
import torch
from torch import Tensor
from rubix.classes.cube import Rubix
from rubix.functions.solver_strategies import STRATEGY

class Solver:

    """
    Solver manages the iterative optimization of solutions using evolutionary principles.

    Attributes:
        seed (torch.Tensor): The initial state for generating solutions.
        window (tuple[int, int]): The dimensions or bounds used during solution evaluation.
        n (int): Number of individuals in each generation.
        epochs (int): Total number of epochs to run the optimization.
    """

    def __init__(
        self,
        seed: Tensor,
        cost_params: Dict[str, Tensor],
        layout_params: Dict[str, Any],
        solver_params: Dict[str, Any]

    ) -> None:
        
        """
        Initializes the Solver with the given parameters.

        Args:
            seed (torch.Tensor): The initial state for generating solutions.
            weights (torch.Tensor): The weights for the optimization process.
            window (Tuple[int, int]): Bounds for evaluating solutions.
            column_indices (torch.Tensor): The column indices for constraints.
            kwargs (dict): Additional keyword arguments, including 'n' and 'epochs'.
        """

        self.solver_params = solver_params

        # Prime the Solution constructor class
        Rubix.class_setup(seed, cost_params, layout_params)

    def solve(self) -> Rubix:
        
        """
        Executes the optimization process using the selected solver strategy.

        This method initializes the solution population and delegates the entire
        optimization procedure to the chosen strategy function. The strategy function
        is responsible for managing iterations, convergence criteria, and returning
        the best found solution.

        Returns:
            Rubix: The Rubix instance representing the best solution found by the strategy.
        """

        strategy_fn = STRATEGY[self.solver_params["strategy"]]

        solution, solution_history = strategy_fn(
            Rubix.initialize(),
            self.solver_params
        )

        # Parse history to list
        solution_history = torch.stack(solution_history).numpy().tolist()
    
        return solution, solution_history


