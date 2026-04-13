from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_random_forest(X, y):
    """
    Train a Random Forest regression model.

    Args:
        X (pd.DataFrame): Features for training.
        y (pd.Series): Target variable.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Random Forest Model Mean Squared Error: {mse:.2f}")

    return model
