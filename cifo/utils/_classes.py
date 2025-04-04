import pandas as pd

class Preprocessor:
    def __init__(self, path, weights, on, partitions):
        self.data = path
        self.parition_on = on
        self.partitions = partitions

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
    
    def get_weights():
        # Reset the indices to obtain a unique Id
        data.reset_index(inplace=True)

        # Define the custom order based on the PARTITIONS dictionary keys
        custom_order = list(PARTITIONS.keys())

        # Set the custom order in the 'position' column using pd.Categorical
        data['Position'] = pd.Categorical(
            data['Position']
            , categories=custom_order
            , ordered=True
            )

        # Sort data by Position
        data = data.sort_values(by='Position').to_numpy()

