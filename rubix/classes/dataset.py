# rubix/classes/dataset.py
""" 
dataset.py - This module defines the DataSet dataclass, which centralizes the management 
of raw data, processed DataFrames, and matrix representations.
"""


import numpy as np
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class DataSet:

    """
    DataSet is an immutable container for structured and unstructured data used 
    throughout the application.

    Attributes:
        constructors (Dict[Any, Any]): A mapping of constructor references or metadata.
        dataframe (pd.DataFrame): Tabular representation of the data.
        matrix (np.ndarray): Numpy matrix form of the data for numerical operations.
    """

    constructors : Dict[Any,Any]
    dataframe : pd.DataFrame 
    matrix : np.ndarray


    def __repr__(
        self
    ) -> str:

        """
        Custom string representation of the DataSet for easier inspection and debugging.

        Returns:
    def __repr__(self):a summary of constructors, 
                 first rows of the DataFrame, and a preview of the matrix.
        """

        return f"""DataSet(
            constructors: {self.constructors},
            dataframe: {self.dataframe.head()},
            matrix: {self.matrix[:5]}
        )"""