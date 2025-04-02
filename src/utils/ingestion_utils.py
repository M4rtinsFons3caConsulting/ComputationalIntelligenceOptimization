import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from constants import COUNTRIES

#### ------------------ PROCESSING FUNCTIONS ------------------ ####
def concatenate_csv(directory_path: str) -> pd.DataFrame:
    """Concatenates all CSV files in a specified directory into a single DataFrame.

    This function reads all CSV files in the given directory, concatenates them
    into one DataFrame, and returns the result. It assumes that all CSV files
    have the same structure (i.e., same columns).

    Args:
        directory_path (str): The path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data from all CSV files.

    Raises:
        FileNotFoundError: If no CSV files are found in the specified directory.

    Examples:
        concatenate_csv('/path/to/directory') -> pd.DataFrame
    """
    abs_dir_path = Path(directory_path)
    csv_files = list(abs_dir_path.glob('*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory_path}")

    df_list: List[pd.DataFrame] = [pd.read_csv(file) for file in csv_files]
    
    return pd.concat(df_list, ignore_index=True)


def to_snake_case(iterable: Union[List[str], str]) -> Union[str, List[str]]:
    """
    Convert a string or each element of a list of strings to snake_case.

    This function processes the input by converting each string in the input
    (whether it is a single string or a list of strings) to snake_case. It:
    - Strips leading/trailing whitespace
    - Removes unwanted characters, leaving only alphanumeric characters and underscores
    - Replaces spaces and hyphens with underscores
    - Converts camelCase and PascalCase to snake_case
    - Converts the result to lowercase

    Args:
        iterable (Union[List[str], str]): A string or a list of strings to be converted to snake_case.
            If a single string is provided, it returns the snake_case version of that string.
            If a list is provided, it returns a list of strings, each converted to snake_case.

    Returns:
        Union[str, List[str]]: 
            - If the input is a single string, returns the snake_case version of that string.
            - If the input is a list of strings, returns a list of strings, each in snake_case.

    Examples:
        to_snake_case(["HelloWorld!"]) -> "hello_world"
        to_snake_case(["convertToSnakeCase@", "AnotherTest"]) -> ["convert_to_snake_case", "another_test"]
        to_snake_case("this is a test!!!") -> "this_is_a_test"
    """
    
    def _parse_to_snake_case(s: str) -> str:
        # Strip leading and trailing spaces
        s = s.strip()
        # Remove undesired symbols (keep only alphanumeric and underscores)
        s = re.sub(r'[^\w\s-]', '', s)
        # Replace spaces and dashes with underscores
        s = re.sub(r'[\s-]+', '_', s)
        # Convert camelCase or PascalCase to snake_case
        s = re.sub(r'([a-z])([A-Z])', r'\1_\2', s)
        # Convert to lowercase
        s = s.lower()
        # And check for usual currency denomination
        return s.replace('us_mil', 'usd_mil')

    # Handle if input is a single string or a list with exactly one string
    if isinstance(iterable, str) or (isinstance(iterable, list) and len(iterable) == 1):
        return _parse_to_snake_case(str(iterable))
    
    # Process each element in the list individually if it's a list with more than one element
    return [_parse_to_snake_case(str(s)) for s in iterable]
