def preprocess_data(data, target_col):
    """
    Preprocess the data by handling missing values and ensuring proper formatting.

    Args:
        data (pd.DataFrame): Raw data loaded from a CSV file.
        target_col (str): Name of the target column for forecasting.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Handle missing values by forward-filling
    data[target_col] = data[target_col].ffill()
    return data
