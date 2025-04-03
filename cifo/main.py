import pandas as pd
from classes.solver import Solver
from constants import PATH, ON, PARTITIONS, WEIGHTS, SOLVER_KWARGS


def load_data(path: str) -> tuple[pd.Series, pd.DataFrame]:
    """
    Load data from an Excel file and extract labels and weights.

    Parameters:
    - path (str): Path to the Excel file.

    Returns:
    - labels (pd.Series): Target labels extracted from the dataset.
    - weights (pd.DataFrame): Feature weights extracted from the dataset.
    """
    try:
        data = pd.read_excel(path, index_col='Unnamed: 0')

        # Extract required columns and return as NumPy arrays
        labels = data[ON].to_numpy()
        weights = data[WEIGHTS].to_numpy()

        return labels, weights

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {path}")
    except KeyError as e:
        raise KeyError(f"Missing required column in dataset: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


def main():
    """Main execution routine."""
    labels, weights = load_data(PATH)

    # Initialize and solve, no need to pass global constants explicitly
    final_solution = Solver(labels, weights, ON, PARTITIONS, SOLVER_KWARGS)
    
    # Print result
    print(final_solution)


if __name__ == "__main__":
    main()
