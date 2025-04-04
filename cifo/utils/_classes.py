import pandas as pd

class Preprocessor:
    def __init__(self, data, weights, on, ):

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

            return data
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File not found at {path}")
        except KeyError as e:
            raise KeyError(f"Missing required column in dataset: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
        

processed_data = Processor