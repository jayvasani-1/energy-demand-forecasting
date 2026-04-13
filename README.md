# Short-Term Energy Demand Forecasting

## 📌 Overview
This project focuses on **short-term energy demand forecasting** using **Random Forest (RF) and Long Short-Term Memory (LSTM)** models. It processes historical energy consumption data to predict future energy demand.

## 📂 Project Structure
📁 energy_forecasting/ │── 📁 data/ # Data directory │ ├── 📁 raw/ # Raw dataset files │ ├── 📁 features/ # Processed feature files │ ├── 📁 outputs/ # Model outputs & results │── 📁 scripts/ # Code files │ ├── eda.py # Exploratory Data Analysis │ ├── preprocessing.py # Data Preprocessing │ ├── feature_engineering.py # Feature Engineering │ ├── train_model.py # Train Random Forest │ ├── train_lstm.py # Train LSTM │ ├── forecast.py # Forecasting with trained models │ ├── utils.py # Utility functions │── requirements.txt # Required dependencies │── main.py # Main execution script │── README.md # Project documentation



## 🚀 Installation & Setup

1. **Clone the repository**:
   ```
 
Create a virtual environment (optional but recommended):


python -m venv venv
source venv/bin/activate  # On macOS/Linux
# Short-Term Energy Demand Forecasting

## 📌 Overview
This project focuses on **short-term energy demand forecasting** using **Random Forest (RF) and Long Short-Term Memory (LSTM)** models. It processes historical energy consumption data to predict future energy demand.

## 📂 Project Structure
📁 energy_forecasting/ │── 📁 data/ # Data directory │ ├── 📁 raw/ # Raw dataset files │ ├── 📁 features/ # Processed feature files │ ├── 📁 outputs/ # Model outputs & results │── 📁 scripts/ # Code files │ ├── eda.py # Exploratory Data Analysis │ ├── preprocessing.py # Data Preprocessing │ ├── feature_engineering.py # Feature Engineering │ ├── train_model.py # Train Random Forest │ ├── train_lstm.py # Train LSTM │ ├── forecast.py # Forecasting with trained models │ ├── utils.py # Utility functions │── requirements.txt # Required dependencies │── main.py # Main execution script │── README.md # Project documentation



## 🚀 Installation & Setup

1. **Clone the repository**:
   ```
 
Create a virtual environment (optional but recommended):


python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

# Install dependencies:
pip install -r requirements.txt

# Running the Project
1.️Exploratory Data Analysis (EDA)
bash
Copy
Edit
python scripts/eda.py
Generates basic statistics and visualizations of the dataset.

2.️Data Preprocessing & Feature Engineering

python scripts/preprocessing.py
python scripts/feature_engineering.py
Handles missing values, normalizes data, and extracts time-based features.

3.️Train Models
Train Random Forest

python scripts/train_model.py
Train LSTM

python scripts/train_lstm.py
Trains and saves the trained models in outputs/models/.

4.️Generate Forecast

python scripts/forecast.py
Generates predictions using trained models.
📁 Outputs
Forecasting Plots: outputs/plots/
Trained Models: outputs/models/
Predictions: outputs/predictions.csv
⚠️ Troubleshooting
Ensure all dependencies are installed (pip install -r requirements.txt).
Verify the dataset path in data/raw/.
Check for missing directories (outputs/ folder must exist before running).
     # On Windows

# Install dependencies:
pip install -r requirements.txt

# Running the Project
1.️Exploratory Data Analysis (EDA)
bash
Copy
Edit
python scripts/eda.py
Generates basic statistics and visualizations of the dataset.

2.️Data Preprocessing & Feature Engineering

python scripts/preprocessing.py
python scripts/feature_engineering.py
Handles missing values, normalizes data, and extracts time-based features.

3.️Train Models
Train Random Forest

python scripts/train_model.py
Train LSTM

python scripts/train_lstm.py
Trains and saves the trained models in outputs/models/.

4.️Generate Forecast

python scripts/forecast.py
Generates predictions using trained models.
📁 Outputs
Forecasting Plots: outputs/plots/
Trained Models: outputs/models/
Predictions: outputs/predictions.csv
⚠️ Troubleshooting
Ensure all dependencies are installed (pip install -r requirements.txt).
Verify the dataset path in data/raw/.
Check for missing directories (outputs/ folder must exist before running).
