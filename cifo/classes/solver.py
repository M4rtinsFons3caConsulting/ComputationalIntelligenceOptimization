import numpy as np
from typing import Any
from cifo.classes.solution import Solution

class Solver:
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

    def solve(self):

        print(f"Running for {self.epochs} epochs. Creating {self.n} individuals in each epoch, and computing the mean min fitness.")
        
        fitness_array = np.zeros(self.epochs)

        population = Solution.initialize(self.seed, self.n)

        # DO stuff, 

        # Return solution
        return min(population)
