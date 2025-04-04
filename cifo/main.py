from constants import PATH, LABELS, LABELS, WEIGHTS, SOLVER_KWARGS
from utils._classes import Preprocessor as p
from classes.solver import Solver


def main():
    """Main execution routine."""
    # Load the data
    data, shape = p.process_data(PATH, LABELS, LABELS, WEIGHTS)

    # Initialize and solve
    final_solution = Solver(data, shape, SOLVER_KWARGS)
    
    # Print result
    print(final_solution)


if __name__ == "__main__":
    main()
    