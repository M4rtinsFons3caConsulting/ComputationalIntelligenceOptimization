from typing import Any, Dict
from rubix.classes.cube import Rubix
from torch import Tensor

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

    def solve(self) -> float:
        """
        Runs the optimization for a fixed number of epochs and returns the best fitness score found.

        Returns:
            float: The fitness value of the best solution in the final population.
        """
        print(f"Running for {self.solver_params['epochs']} epochs.")

        # Initialize the population (cube)
        cube = Rubix.initialize()
        best_cube = cube

        for epoch in range(self.solver_params['epochs']):
            print(f"Epoch {epoch + 1}/{self.solver_params['epochs']}...")
            
            # Get new mappings and apply permutation
            roll_map, swap_map, mode_map = cube.get_mappings()
            
            # Compute a new cube
            new_cube = cube.rubix_permute(cube.solutions, roll_map, swap_map, mode_map)

            # Update the best solution if fitness improves
            if new_cube < best_cube:
                best_cube = new_cube

        return best_cube



