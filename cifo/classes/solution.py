import numpy as np
import pandas as pd

class Solution:
    def __init__(self, shape: tuple, data: pd.DataFrame) -> None:
        self.solution = np.empty(shape)
        