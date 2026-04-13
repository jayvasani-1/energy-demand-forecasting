import pandas as pd
import matplotlib.pyplot as plt

def run_eda():
    # Load data
    data = pd.read_csv('data/raw/AEP_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')

    # Basic statistics
    print(data.describe())

    # Plot energy demand trends
    plt.figure(figsize=(12, 6))
    data['MW'].plot(title='Energy Demand over Time', xlabel='Datetime', ylabel='Energy Demand (MW)')
    plt.show()

if __name__ == "__main__":
    run_eda()
