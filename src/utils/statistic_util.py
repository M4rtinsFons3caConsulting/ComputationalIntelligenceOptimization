from statsmodels.tsa.stattools import adfuller, coint

# Function for feature selection based on cointegration
def cointegration_test(target_series, features):
    """
    Perform feature selection using cointegration tests.
    
    Parameters:
    target_series (pd.Series): The target time series variable.
    features (pd.DataFrame): DataFrame containing the predictor features (time series).
    
    Returns:
    pd.DataFrame: Reduced features after cointegration.
    """
    
    # Step 1: Check for stationarity of each feature (ADF test)
    def check_stationarity(series):
        result = adfuller(series)
        return result[1]  # p-value
    
    # List of stationary features (non-stationary are eligible for cointegration testing)
    stationary_features = [col for col in features.columns if check_stationarity(features[col]) > 0.05]
    
    print(f"Stationary Features: {stationary_features}")
    
    # Step 2: Perform Cointegration Test (Engle-Granger)
    cointegrated_features = []
    for col in stationary_features:
        score, p_value, _ = coint(target_series, features[col])
        if p_value < 0.05:  # If cointegration exists (p-value < 0.05)
            cointegrated_features.append(col)
    
    print(f"Cointegrated Features: {cointegrated_features}")
    
    return features[cointegrated_features]