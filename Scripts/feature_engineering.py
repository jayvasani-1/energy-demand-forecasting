def extract_features(data, target_col):
    """
    Extract features for the machine learning model, including time-based and lag features.

    Args:
        data (pd.DataFrame): Preprocessed data.
        target_col (str): Name of the target column for forecasting.

    Returns:
        pd.DataFrame: Data with extracted features.
    """
    # Add time-based features
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month

    # Add lag features for target column
    data['lag_1'] = data[target_col].shift(1)
    data['lag_2'] = data[target_col].shift(2)
    data['lag_24'] = data[target_col].shift(24)  # Lag for same hour on the previous day

    # Drop rows with NaN values caused by lag features
    return data.dropna()
