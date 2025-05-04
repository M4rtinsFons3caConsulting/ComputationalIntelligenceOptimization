""" The DataSet dataclass centralizes data management. """

from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class DataSet:
    constructors : Dict[Any,Any]
    dataframe : pd.DataFrame 
    matrix : np.ndarray

    def __repr__(self):
        return f"""DataSet(
            constructors: {self.constructors},
            dataframe: {self.dataframe.head()},
            matrix: {self.matrix[:5]}
        )"""