from typing import List
import numpy as np
from numba import types, njit

@njit
def permute_blocks(seed_matrix: np.ndarray, constraints: np.ndarray, 
                    costs_array: np.ndarray, ability_array: np.ndarray, n: int) -> np.ndarray:
    counter = 0
    solution_array = np.empty((n, *seed_matrix.shape), dtype=np.int64)
    fitness_array = np.empty(n, dtype=np.float64)

    while counter < n:
        ### ---- Create the random solution from the seed ----###
        curr_solution = seed_matrix.copy()
        start_col = 0

        for block_size in constraints:
            end_col = start_col + block_size
            curr_solution[:, start_col:end_col] = curr_solution[np.random.permutation(curr_solution.shape[0]), start_col:end_col]
            start_col = end_col

        ### --- Validate the solution, and calculate its fitness --- ###
        flat_idx = curr_solution.ravel()
        reshaped_ability = ability_array[flat_idx].reshape(curr_solution.shape)
        reshaped_costs = costs_array[flat_idx].reshape(curr_solution.shape)

        ability_row_sums = np.empty(curr_solution.shape[0], dtype=np.int64)
        valid = True
        
        for i in range(curr_solution.shape[0]):
            row_costs = reshaped_costs[i]
            if np.sum(row_costs) > 750:
                valid = False
                break

            # Row sum for ability array
            ability_row_sums[i] = np.sum(reshaped_ability[i])

        if not valid:
            continue
        
        # Now compute the standard deviation of the row sums
        ability_row_mean = np.mean(ability_row_sums)
        variance = np.mean((ability_row_sums - ability_row_mean) ** 2)
        curr_fitness = np.sqrt(variance)

        # And assign to the arrays
        if valid:
            solution_array[counter] = curr_solution
            fitness_array[counter] = curr_fitness
            counter += 1
    
    return solution_array, fitness_array


class Solution:   
    abilities_array = None
    costs_array = None
    constraints = None  # Add constraints as class-level variable

    def __init__(self, solution_array: np.ndarray, fitness: int) -> None:
        self.solution = solution_array
        self.fitness = fitness

    def roll(self, shift: int) -> None:
        """Roll the solution array by 'shift' positions using np.roll, keeping fitness unchanged."""
        self.solution = np.roll(self.solution, shift)

    def update_fitness(self) -> None:
        """Update fitness to the last value of the solution array."""
        # Use the last element dynamically instead of storing last_index
        self.fitness = self.solution[-1]  # Access the last element directly

    def __lt__(self, other):
        """Compare the fitness values between self and another Solution."""
        return self.fitness < other.fitness

    def __le__(self, other):
        """Compare the fitness values between self and another Solution."""
        return self.fitness <= other.fitness

    def __gt__(self, other):
        """Compare the fitness values between self and another Solution."""
        return self.fitness > other.fitness

    def __ge__(self, other):
        """Compare the fitness values between self and another Solution."""
        return self.fitness >= other.fitness

    def __eq__(self, other):
        """Check if the fitness values between self and another Solution are equal."""
        return self.fitness == other.fitness

    @classmethod
    def set_constraints(cls, constraints: np.ndarray) -> None:
        cls.constraints = constraints

    @classmethod
    def set_weights(cls, weights: np.ndarray) -> None:
        cls.abilities_array, cls.costs_array = weights[:, 0], weights[:, 1]

    @classmethod
    def initialize(cls: type["Solution"], seed_matrix: np.ndarray, n: int = 100) -> List["Solution"]:
        solutions, fits = cls.permute_blocks(
              seed_matrix.astype(np.int64)
            , cls.constraints.astype(np.int64)
            , cls.abilities_array.astype(np.int64)
            , cls.costs_array.astype(np.int64)
            , n
        )

        return [cls(solutions[i], fits[i]) for i in range(n)]

# Trigger compilation for the permute_blocks function
permute_blocks.compile((
    types.Array(types.int64, 2, 'C'),  # seed_matrix (2D int64 array in C-contiguous order)
    types.Array(types.int64, 1, 'C'),  # constraints (1D int64 array in C-contiguous order)
    types.Array(types.int64, 1, 'C'),  # costs_array (1D int64 array in C-contiguous order)
    types.Array(types.int64, 1, 'C'),  # ability_array (1D int64 array in C-contiguous order)
    types.int64                        # n (scalar int64)
))
