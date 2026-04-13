def generate_forecast(model, X):
    """
    Generate forecasts using the trained model.

    Args:
        model (RandomForestRegressor): Trained model.
        X (pd.DataFrame): Features for prediction.

    Returns:
        np.array: Forecasted values.
    """
    # Predict values using the trained model
    forecast = model.predict(X)
    return forecast
