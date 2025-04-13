import numpy as np
from classes.solution import Solution

class Solver:
    def __init__(self, seed, weights, window, constraints, kwargs):
        # Labels, weights and args
        self.kwargs = kwargs
        
        import time
        # Start timing
        start_time = time.perf_counter()

        Solution.set_weights(weights)
        population = Solution.initialize(seed, np.array(list(constraints.values())), self.kwargs['n'])
        
        # End timing
        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        
        print(min(population, key=lambda x: x.fitness))

        print(f"Execution Time: {elapsed_time:.6f} seconds")
