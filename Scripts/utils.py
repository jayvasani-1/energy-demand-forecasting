import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data with datetime index.
    """
    data = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
    return data

def save_model(model, file_path):
    """
    Save the trained model to a file, ensuring the directory exists.

    Args:
        model: Trained model (e.g., RandomForestRegressor).
        file_path (str): Path to save the model.
    """
    # Ensure the directory exists before saving
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def plot_forecast(dates, actual, forecast, save_path):
    """
    Plot the forecasted values against actual values.

    Args:
        dates (pd.DatetimeIndex): Dates for the forecast.
        actual (pd.Series): Actual values.
        forecast (np.array): Forecasted values.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(dates, forecast, label='Forecast', color='orange', linestyle='--')
    plt.xlabel('Datetime')
    plt.ylabel('Energy Consumption (MW)')
    plt.title('Actual vs Forecasted Energy Consumption')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Ensure the directory exists before saving the plot
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(save_path)
    plt.close()
