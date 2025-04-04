import pandas as pd
from utils._classes import Preprocessor as p
from classes.solver import Solver
from constants import PATH, ON, PARTITIONS, WEIGHTS, SOLVER_KWARGS

def main():
    """Main execution routine."""
    data = p.process_data(PATH, ON, PARTITIONS, WEIGHTS)

    # Initialize and solve, no need to pass global constants explicitly
    final_solution = Solver(labels, weights, ON, PARTITIONS, SOLVER_KWARGS)
    
    # Print result
    print(final_solution)


if __name__ == "__main__":
    main()
