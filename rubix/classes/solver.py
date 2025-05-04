# rubix/classes/solver.py

"""
solver.py -This module defines the Solver class, which orchestrates the evolutionary process 
for optimizing a solution using given weights, constraints, and a seed state.
"""


import numpy as np
from typing import Any
from rubix.classes.solution import Solution


class Solver:

    """
    Solver manages the iterative optimization of solutions using evolutionary principles.

    Attributes:
        seed (np.ndarray): The initial state for generating solutions.
        window (tuple[int, int]): The dimensions or bounds used during solution evaluation.
        n (int): Number of individuals in each generation.
        epochs (int): Total number of epochs to run the optimization.
    """

    def __init__(
        self: type['Solver'],
        seed: np.ndarray,
        weights: np.ndarray,
        window: tuple[int, int],
        constraints: np.ndarray,
        kwargs: dict[str, Any]
    ) -> None:
        self.seed = seed
        self.window = window
        self.n = kwargs['n']
        self.epochs = kwargs['epochs']

        Solution.set_constraints(constraints)
        Solution.set_weights(weights)


    def solve(
        self
    ) -> Solution.fitness:

        """
        Runs the optimization for a fixed number of epochs and returns the best fitness score found.

        Returns:
            float: The fitness value of the best solution in the final population.
        """

        print(f"Running for {self.epochs} epochs. Creating {self.n} individuals in each epoch, and computing the mean min fitness.")
        

        population = Solution.initialize(self.seed, self.n)

        return min(population).fitness
