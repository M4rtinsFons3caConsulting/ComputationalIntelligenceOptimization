from constants import PATH, LABELS, CONSTRAINTS, WEIGHTS, SOLVER_KWARGS
from utils._classes import Preprocessor as p
from classes.solver import Solver

# Optional: For debugging / plotting
from utils._functions import call_seaborn


def main():
    """Main execution routine."""
    
    # Preprocessing step
    seed_matrix, feature_matrix, fitness_window = p.process_data(
        path=PATH,
        label_col=LABELS,
        partitions=CONSTRAINTS,
        feature_cols=WEIGHTS
    )

    # Solve the optimization problem
    solution = Solver(seed_matrix, feature_matrix, SOLVER_KWARGS)

    # Debug / visualize if needed
    # call_seaborn(solution.result)  # Uncomment if result is visualizable

    # Optional: print output or log summary
    # print("Solution:", solution)

if __name__ == "__main__":
    main()
