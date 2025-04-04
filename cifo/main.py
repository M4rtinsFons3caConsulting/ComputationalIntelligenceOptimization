import pandas as pd
from classes.solver import Solver
from constants import PATH, ON, PARTITIONS, WEIGHTS, SOLVER_KWARGS

def main():
    """Main execution routine."""
    data = load_data(PATH)



    # Initialize and solve, no need to pass global constants explicitly
    final_solution = Solver(labels, weights, ON, PARTITIONS, SOLVER_KWARGS)
    
    # Print result
    print(final_solution)


if __name__ == "__main__":
    main()
