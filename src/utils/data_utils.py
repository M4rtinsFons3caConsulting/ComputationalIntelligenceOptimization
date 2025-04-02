import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from IPython import get_ipython
from pathlib import Path
from typing import List, Dict, Tuple, Optional


 #### ------------------ ANALYSIS FUNCTIONS ------------------ ####
def get_aggregations(
    _data: pd.DataFrame, 
    _type: str = 'metric',  # Options: 'metric', 'categorical', or 'both'
    selected: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregates selected columns of a DataFrame based on specified metric functions.

    Args:
        _data (pd.DataFrame): The input DataFrame.
        _type (str): Type of aggregation ('metric', 'categorical', or 'both').
        selected (List[str]): List of columns to aggregate.

    Returns:
        pd.DataFrame: Aggregated results.
    """
    from scipy.stats import gmean, hmean

    #### ------------------ AGGREGATION FUNCTIONS ------------------ ####
    def mode(x: pd.Series) -> Optional[float]:
        """Returns the mode of a series. If multiple modes exist, returns the first one."""
        return x.mode().iloc[0] if not x.mode().empty else None

    def _25(x: pd.Series) -> float:
        """Returns the 25th percentile of a series."""
        return x.quantile(0.25)

    def _75(x: pd.Series) -> float:
        """Returns the 75th percentile of a series."""
        return x.quantile(0.75)

    def _90(x: pd.Series) -> float:
        """Returns the 90th percentile of a series."""
        return x.quantile(0.90)

    def _95(x: pd.Series) -> float:
        """Returns the 95th percentile of a series."""
        return x.quantile(0.95)

    def _98(x: pd.Series) -> float:
        """Returns the 98th percentile of a series."""
        return x.quantile(0.98)

    def _gmean(x: pd.Series) -> float:
        """Returns the geometric mean of a series."""
        if (x < 0).sum() > 0:
            warnings.warn("The geometric mean can only be calculated for series of strictly positive values", category=UserWarning)
            return np.nan
        else:
            return gmean(x)

    def _hmean(x: pd.Series) -> float:
        """Returns the harmonic mean of a series."""
        if (x <= 0).sum() > 0:
            warnings.warn("The harmonic mean can only be calculated for series of strictly positive values", category=UserWarning)
            return np.nan
        else:
            return hmean(x)

    # Metric functions
    metric_functions = [
        'sum', 'mean', 'std', 'var', 'skew', 'kurt', 'min', 
        _25, 'median', _75, _90, _95, _98, 'max', mode, _gmean, _hmean
    ]

    # Categorical functions
    categorical_functions = ['count', mode]

    if not selected:
        selected = _data.columns

    if _type == 'metric':
        agg_dict = {col: metric_functions for col in selected if pd.api.types.is_numeric_dtype(_data[col])}
    
    elif _type == 'categorical':
        agg_dict = {col: categorical_functions for col in selected if not pd.api.types.is_numeric_dtype(_data[col])}

    else:
        agg_dict = {}
        for col in selected:
            if pd.api.types.is_numeric_dtype(_data[col]):
                agg_dict[col] = metric_functions
            else:
                agg_dict[col] = categorical_functions

    return _data[selected].agg(agg_dict).round(2).T


def sturges_bins(data: pd.DataFrame, column_name: str) -> int:
    """
    Calculates the number of bins for a histogram using Sturges' Rule.

    Args:
        data (pd.DataFrame): The dataset containing the column.
        column_name (str): The name of the column to calculate the number of bins for.

    Returns:
        int: The number of bins calculated using Sturges' Rule.
    """
    n: int = len(data[column_name])
    return int(np.ceil(np.log2(n) + 1))

def compare_with_target(data: pd.DataFrame, variable: str, target: str) -> pd.DataFrame:
    """
    Compares a categorical variable with a target variable and computes the percentage distribution.

    Args:
        data (pd.DataFrame): The dataset containing the variables.
        variable (str): The categorical variable for grouping.
        target (str): The target variable to compare.

    Returns:
        pd.DataFrame: A DataFrame containing the grouped counts and percentages.
    """
    # Group by the specified variable and target, then count occurrences
    group_counts = data.groupby([variable, target]).size().reset_index(name='Count')

    # Calculate total counts for each unique value of the specified variable
    total_counts = data.groupby(variable).size().reset_index(name='Total')

    # Merge total counts with group counts
    merged_counts = pd.merge(group_counts, total_counts, on=variable)

    # Calculate percentages
    merged_counts['Percentage'] = ((merged_counts['Count'] / merged_counts['Total']) * 100).round(2)

    # Drop the 'Total' column
    merged_counts = merged_counts.drop(columns=['Total'])

    return merged_counts

def show_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Shows missing values, data types, and unique counts for each column in the DataFrame.

    Args:
        data (pd.DataFrame): The dataset to analyze.

    Returns:
        pd.DataFrame: A summary DataFrame containing missing value statistics.
    """
    variables: List[str] = []
    dtypes: List[str] = []
    count: List[int] = []
    unique: List[int] = []
    missing: List[int] = []
    pc_missing: List[float] = []

    for item in data.columns:
        variables.append(item)
        dtypes.append(data[item].dtype)
        count.append(len(data[item]))
        unique.append(len(data[item].unique()))
        missing_count = data[item].isna().sum()
        missing.append(missing_count)
        pc_missing.append(round((missing_count / len(data[item])) * 100, 2))

    return pd.DataFrame({
        'variable': variables, 
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing': missing, 
        'pc_missing': pc_missing,
    })

def compute_percentile_bounds(
    reference_data: pd.DataFrame, 
    filters: Dict[str, Tuple[Optional[float], Optional[float]]]
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Compute percentile bounds for the given filters using the reference dataset.

    Args:
        reference_data (pd.DataFrame): DataFrame used to compute the percentiles.
        filters (Dict[str, Tuple[Optional[float], Optional[float]]]): Dictionary of column names 
            and (lower_percentile, upper_percentile).

    Returns:
        Dict[str, Tuple[Optional[float], Optional[float]]]: Dictionary with column names and the 
        corresponding (lower_bound, upper_bound).
    """
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    for col, percentiles in filters.items():
        if col not in reference_data.columns:
            raise ValueError(f"Column '{col}' not found in the reference DataFrame.")

        lower_percentile, upper_percentile = percentiles
        lower_bound = reference_data[col].quantile(lower_percentile) if lower_percentile is not None else None
        upper_bound = reference_data[col].quantile(upper_percentile) if upper_percentile is not None else None
        bounds[col] = (lower_bound, upper_bound)

    return bounds
