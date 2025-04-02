import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from IPython import get_ipython
from pathlib import Path
from typing import List, Dict, Tuple, Optional

#### ------------------ DISPLAY & STYLE SETTINGS ------------------ ####
def set_display() -> None:
    """Set pandas display options and apply seaborn style."""
    # Pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_seq_items', None)
    pd.set_option('display.max_colwidth', None)

    # Apply Seaborn style for better visuals
    sns.set(style="white")

    # Apply Retina display setting for Jupyter notebooks (if applicable)
    try:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    except NameError:
        pass  # Ignore if not in Jupyter Notebook


def plot_predictions(data, predictions, target_column):
    """
    Function to plot actual vs. predicted values for time series forecasting.

    Parameters:
    - data (pd.DataFrame): The dataset containing actual values with a DateTime index.
    - predictions (pd.Series): The predicted values with corresponding future dates as the index.
    - target_column (str): The name of the actual values column in `data`.

    Returns:
    - None (Displays a plot)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    
    # Plot actual values
    plt.plot(data[target_column], label='Actual', linestyle='solid', color='blue')
    
    # Plot predicted values
    plt.plot(predictions, label='Predicted', linestyle='dashed', color='red')

    # Add title and labels
    plt.title(f'Predicted vs Actual for {target_column}')
    plt.xlabel('Time')
    plt.ylabel('Index Value')
    
    # Show legend and grid
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


# def plot_predictions(df_val, df_4cast, model):
#     # Plot forecast
#     forecast_plot = model.plot(df_4cast)

#     # Add vertical line at the last training date
#     axes = forecast_plot.gca()
#     last_training_date = df_4cast['ds'].iloc[-test_set_units]
#     axes.axvline(x=last_training_date, color='red', linestyle='--', label='Training End')

#     # Plot true data
#     plt.plot(self.df_val['Date'], self.df_val[self.target], 'ro', markersize=3, label='True Test Data')

#     plt.title(f'Forecast for {self.target}')
#     plt.legend()
#     plt.show()
