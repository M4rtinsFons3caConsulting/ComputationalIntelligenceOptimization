# rubix/classes/solver.py


"""
solver.py -This module defines the Solver class, which orchestrates the evolutionary process 
for optimizing a solution using given weights, constraints, and a seed state.
"""


import numpy as np
from numba import njit
from typing import Any, Tuple
from rubix.classes.solution import Solution


@njit
def rubix_permute(
    cube: np.ndarray,  # The cube of matrices being passed.
    cube_shape: Tuple[int,int,int], # The shape of the data cube.
    roll_map: np.ndarray, # The mapping of the columns to roll.
    swap_map:np.ndarray, # The mapping of the values to swap.
    mode_map: np.ndarray, # The mapping of the modality of a given matrix's transformation. 
) -> np.ndarray:
    
    """
    This piece of code is the soul of this genetic algorithm implementation. The population is treated like a data Cube, and the 
    column rolling and block swapping is applied to each individual.
    """

    new_cube = cube.copy()
    n_matrices, n_rows, n_cols = cube_shape

    for m in range(n_matrices):
        
        # swap only
        if mode_map[m] == 0: 
            for col in range(n_cols):
                if swap_map[col, 0]:
                    cube[m, swap_map[col, 1], col], cube[m, swap_map[col, 1] + 1, col] = \
                        cube[m, swap_map[col, 1] + 1, col], cube[m, swap_map[col, 1], col]

        # roll only
        elif mode_map[m] == 1:  
            for col in range(n_cols):
                if roll_map[col, 0]:
                    shift = roll_map[col, 1]
                    for row in range(n_rows):
                        cube[m, row, col] = cube[m, (row - shift) % n_rows, col]

        # swap then roll
        elif mode_map[m] == 2:  
            for col in range(n_cols):
                if swap_map[col, 0]:
                    cube[m, swap_map[col, 1], col], cube[m, swap_map[col, 1] + 1, col] = \
                        cube[m, swap_map[col, 1] + 1, col], cube[m, swap_map[col, 1], col]

            for col in range(n_cols):
                if roll_map[col, 0]:
                    shift = roll_map[col, 1]
                    for row in range(n_rows):
                        cube[m, row, col] = cube[m, (row - shift) % n_rows, col]

        # roll then swap
        elif mode_map[m] == 3:  
            for col in range(n_cols):
                if roll_map[col, 0]:
                    shift = roll_map[col, 1]
                    for row in range(n_rows):
                        cube[m, row, col] = cube[m, (row - shift) % n_rows, col]

            for col in range(n_cols):
                if swap_map[col, 0]:
                    cube[m, swap_map[col, 1], col], cube[m, swap_map[col, 1] + 1, col] = \
                        cube[m, swap_map[col, 1] + 1, col], cube[m, swap_map[col, 1], col]

    return new_cube


@njit
def update_cube(
    incumbent_cube: np.ndarray, 
    challenger_cube: np.ndarray,
    cube_shape: np.ndarray, 
) -> None:
    
    # Initialize the cube shape
    n_matrices, n_rows, _ = cube_shape
    
    # Initialize an empty fitness_array 
    fitness_array = np.zeros(cube_shape[0])

    # Evaluate the fitness of each challenger matrix
    for m in range(n_matrices):
        row_means = np.empty(n_rows)
        for i in range(n_rows):
            row_means[i] = np.mean(challenger_cube[m, i])
        fitness_array[m] = np.std(row_means)

    # Create a list of winning solutions 
    winners = np.empty(len(n_matrices))

    # Set an auxiliary counter
    counter = 0

    for m in range(n_matrices):
        if fitness_array[m] > incumbent_cube[m].fitness:
            winners[counter] = m, fitness_array[m] 
        else:
            pass             
    
    # 

    return fitness_array


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
        column_indices: np.ndarray,
        kwargs: dict[str, Any]
    ) -> None:
        
        # Ingest attributes
        self.seed = seed
        self.window = window
        self.n = kwargs['n']
        self.epochs = kwargs['epochs']

        # Prime the Solution constructor class
        Solution.set_constraints(column_indices) # FIXME: if we are calculating block_starts that is all we need constraints for consider creating this as part of loader.
        Solution.set_weights(weights)


    def mutate(
        self, 
        population,
        col_mapping: np.ndarray, 
        roll_map: np.ndarray
    ) -> None:

        # Stack all matrices in the population into a 3D array
        batch = np.array([solution.solution_matrix for solution in population])
        
        # Apply rolling to the entire batch
        updated_batch = rubix_permute(batch, col_mapping, roll_map)
        
        # Retrieve the updated matrix for each individual solution
        for i, solution in enumerate(self.population):
            solution.matrix = updated_batch[i]
            solution.fitness = solution.get_fitness()  # TODO: SHould I calculate all fitnesses in a njit loop 

    @staticmethod
    def get_mappings(
        cube_shape: Tuple[int, int, int], 
        block_starts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n_matrices, n_rows, n_cols = cube_shape

        # --- Roll Mapping ---
        roll_flags = np.random.rand(n_cols) < 0.5
        roll_shifts = np.random.randint(1, n_rows, size=n_cols, dtype=np.int64)
        roll_map = np.zeros((n_cols, 2), dtype=np.int64)
        roll_map[:, 0] = roll_flags.astype(np.int64)
        roll_map[:, 1] = roll_shifts * roll_flags

        # --- Swap Mapping ---
        swap_flags = np.random.rand(n_cols) < 0.5
        random_block_indices = np.random.randint(0, len(block_starts), size=n_cols)
        swap_lefts = block_starts[random_block_indices]
        valid_swaps = (swap_lefts + 1) < n_rows
        swap_flags &= valid_swaps
        swap_map = np.zeros((n_cols, 2), dtype=np.int64)
        swap_map[:, 0] = swap_flags.astype(np.int64)
        swap_map[:, 1] = swap_lefts * swap_flags

        # --- Mode Mapping ---
        mode_map = np.random.randint(0, 4, size=n_matrices, dtype=np.int64)

        return roll_map, swap_map, mode_map

    def solve(
        self
    ) -> float:

        """
        Runs the optimization for a fixed number of epochs and returns the best fitness score found.

        Returns:
            float: The fitness value of the best solution in the final population.
        """

        print(f"Running for {self.epochs} epochs. Creating {self.n} individuals in each epoch, and computing the mean min fitness.")
        
        population = Solution.initialize(self.seed, self.n)

        while True:
            moves = self.get_mappings(cube_shape=(1, 5, 7), block_starts=self.block_starts)
            population[0].solution_array = rubix_permute(population[0], cube_shape=(1,5,7), *moves)
            
        return min(population).fitness


# A piece of tech for my Master's thesis
#
#
# from qiskit import QuantumCircuit, Aer, transpile, assemble
# import numpy as np

# def rubix_quantum(batch: np.ndarray, col_mapping: np.ndarray, roll_map: np.ndarray, batch_shape: Tuple[int,int,int]) -> np.ndarray:
#     """
#     This quantum implementation will use quantum circuits to replicate
#     the matrix manipulation: column rolling and block swapping in the batch.
#     """
#     # Create a quantum circuit to represent the batch
#     num_qubits = batch_shape[0] * batch_shape[1] * batch_shape[2]  # Flattened size
#     qc = QuantumCircuit(num_qubits)

#     # Flatten the batch and represent it as quantum states
#     # Each qubit represents a part of the matrix, let's use classical encoding
#     for i, value in enumerate(batch.flatten()):
#         if value == 1:
#             qc.x(i)  # Apply the X gate to flip the qubit to 1 if batch value is 1

#     # Apply column mapping and rolling
#     for col in range(batch_shape[2]):
#         if col_mapping[col]:
#             for row in range(batch_shape[1]):
#                 # Row rolling is akin to shifting qubits
#                 roll = roll_map[col]
#                 # Apply controlled rotations to simulate the roll (using qubits as the state)
#                 qc.rz(np.pi * roll, row * batch_shape[2] + col)

#     # Apply swaps (simulating block swaps for matrix manipulations)
#     for m in range(batch_shape[0]):
#         for col in range(batch_shape[2]):
#             if col_mapping[col]:
#                 for row in range(batch_shape[1]):
#                     # Perform a controlled swap (or use multiple swaps) based on batch and roll
#                     # Create a controlled swap gate between rows
#                     qc.cx(row, (row - roll_map[col]) % batch_shape[1])  # Controlled-X (CX) gate

#     # Simulate the circuit execution
#     simulator = Aer.get_backend('statevector_simulator')
#     compiled_circuit = transpile(qc, simulator)
#     result = simulator.run(compiled_circuit).result()

#     # Extract the result as a numpy array (collapse the quantum state to classical bits)
#     statevector = result.get_statevector()
    
#     # Convert back to a matrix representation
#     out = np.array(statevector).reshape(batch_shape)

#     return out

# # Test the function
# batch = np.random.randint(0, 2, (3, 5, 7))  # Example batch
# col_mapping = np.random.choice([0, 1], size=(7,))
# roll_map = np.random.randint(1, 4, size=(7,))
# batch_shape = batch.shape

# result = rubix_quantum(batch, col_mapping, roll_map, batch_shape)
# print(result)
