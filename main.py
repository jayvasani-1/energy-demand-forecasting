import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Scripts.preprocessing import preprocess_data
from Scripts.feature_engineering import extract_features
from Scripts.train_model import train_random_forest
from Scripts.train_lstm import train_lstm
from Scripts.utils import load_data, save_model, plot_forecast
from tensorflow.keras.models import load_model

# ── base paths ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MODELS  = os.path.join(BASE_DIR, "outputs", "models")
OUTPUT_PREDS   = os.path.join(BASE_DIR, "outputs", "predictions")
OUTPUT_PLOTS   = os.path.join(BASE_DIR, "outputs", "visualizations")

os.makedirs(OUTPUT_MODELS, exist_ok=True)
os.makedirs(OUTPUT_PREDS,  exist_ok=True)
os.makedirs(OUTPUT_PLOTS,  exist_ok=True)

# ── datasets ──────────────────────────────────────────────────────────────────
DATASETS = [
    {"file": os.path.join(BASE_DIR, "data", "raw", "AEP_hourly.csv"),       "target": "AEP_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "COMED_hourly.csv"),      "target": "COMED_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "DAYTON_hourly.csv"),     "target": "DAYTON_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "DEOK_hourly.csv"),       "target": "DEOK_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "DOM_hourly.csv"),        "target": "DOM_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "DUQ_hourly.csv"),        "target": "DUQ_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "EKPC_hourly.csv"),       "target": "EKPC_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "FE_hourly.csv"),         "target": "FE_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "NI_hourly.csv"),         "target": "NI_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "PJM_Load_hourly.csv"),   "target": "PJM_Load_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "PJME_hourly.csv"),       "target": "PJME_MW"},
    {"file": os.path.join(BASE_DIR, "data", "raw", "PJMW_hourly.csv"),       "target": "PJMW_MW"},
]


def main():
    results_summary = []

    for dataset in DATASETS:
        file_path  = dataset["file"]
        target_col = dataset["target"]
        ds_name    = os.path.basename(file_path).split("_hourly")[0]

        # ── skip missing files gracefully ─────────────────────────────────────
        if not os.path.exists(file_path):
            print(f"⚠️  File not found, skipping: {file_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {ds_name}  ({target_col})")
        print(f"{'='*60}")

        # ── step 1: load ──────────────────────────────────────────────────────
        raw_data = load_data(file_path)
        print(f"✅ Loaded {len(raw_data):,} rows")
        print(raw_data.head())

        # ── step 2: preprocess ────────────────────────────────────────────────
        processed_data = preprocess_data(raw_data, target_col)
        print(f"✅ Preprocessed — {len(processed_data):,} rows remaining")

        # ── step 3: feature engineering ───────────────────────────────────────
        features_data = extract_features(processed_data, target_col)
        print(f"✅ Features engineered — {features_data.shape[1]} columns")

        # ── step 4: split X / y ───────────────────────────────────────────────
        X = features_data.drop(columns=[target_col])
        y = features_data[target_col]

        # ══════════════════════════════════════════════════════════════════════
        # RANDOM FOREST
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n🌲 Training Random Forest — {ds_name}...")
        rf_model = train_random_forest(X, y)
        print(f"✅ Random Forest trained")

        # save model
        rf_model_path = os.path.join(OUTPUT_MODELS, f"{ds_name}_rf_model.pkl")
        save_model(rf_model, rf_model_path)
        print(f"💾 RF model saved → {rf_model_path}")

        # predict + evaluate
        forecast_rf = rf_model.predict(X)
        rf_rmse = np.sqrt(mean_squared_error(y, forecast_rf))
        rf_mae  = mean_absolute_error(y, forecast_rf)
        print(f"📊 RF  — RMSE: {rf_rmse:,.2f}  |  MAE: {rf_mae:,.2f}")

        # save predictions
        forecast_rf_df = pd.DataFrame({
            "Datetime":    features_data.index,
            "Forecast_MW": forecast_rf
        })
        rf_pred_path = os.path.join(OUTPUT_PREDS, f"{ds_name}_rf_forecast.csv")
        forecast_rf_df.to_csv(rf_pred_path, index=False)
        print(f"💾 RF predictions saved → {rf_pred_path}")

        # save plot
        rf_plot_path = os.path.join(OUTPUT_PLOTS, f"{ds_name}_rf_forecast_plot.png")
        plot_forecast(features_data.index, y, forecast_rf, rf_plot_path)
        print(f"📈 RF plot saved → {rf_plot_path}")

        # ══════════════════════════════════════════════════════════════════════
        # LSTM
        # ══════════════════════════════════════════════════════════════════════
        print(f"\n🧠 Training LSTM — {ds_name}...")
        lstm_model, X_test_seq, y_test_scaled, lstm_scaler = train_lstm(X.values, y.values)
        print(f"✅ LSTM trained")

        # save model
        lstm_model_path = os.path.join(OUTPUT_MODELS, f"{ds_name}_lstm_model.h5")
        lstm_model.save(lstm_model_path)
        print(f"💾 LSTM model saved → {lstm_model_path}")

        # predict + evaluate
        lstm_predictions_scaled = lstm_model.predict(X_test_seq)
        lstm_predictions        = lstm_scaler.inverse_transform(lstm_predictions_scaled)
        y_test                  = lstm_scaler.inverse_transform(y_test_scaled)
        lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
        lstm_mae  = mean_absolute_error(y_test, lstm_predictions)
        print(f"📊 LSTM — RMSE: {lstm_rmse:,.2f}  |  MAE: {lstm_mae:,.2f}")

        # save predictions
        forecast_lstm_df = pd.DataFrame({
            "Datetime":    features_data.index[-len(y_test):],
            "Forecast_MW": lstm_predictions.flatten()
        })
        lstm_pred_path = os.path.join(OUTPUT_PREDS, f"{ds_name}_lstm_forecast.csv")
        forecast_lstm_df.to_csv(lstm_pred_path, index=False)
        print(f"💾 LSTM predictions saved → {lstm_pred_path}")

        # save plot
        lstm_plot_path = os.path.join(OUTPUT_PLOTS, f"{ds_name}_lstm_forecast_plot.png")
        plot_forecast(features_data.index[-len(y_test):],
                      y_test.flatten(), lstm_predictions.flatten(), lstm_plot_path)
        print(f"📈 LSTM plot saved → {lstm_plot_path}")

        # ══════════════════════════════════════════════════════════════════════
        # COMPARISON
        # ══════════════════════════════════════════════════════════════════════
        if rf_rmse < lstm_rmse:
            winner = f"Random Forest (RMSE {rf_rmse:,.2f} vs {lstm_rmse:,.2f})"
        elif lstm_rmse < rf_rmse:
            winner = f"LSTM (RMSE {lstm_rmse:,.2f} vs {rf_rmse:,.2f})"
        else:
            winner = "Tie"

        print(f"\n🏆 Winner for {ds_name}: {winner}")

        results_summary.append({
            "Dataset":   ds_name,
            "RF_RMSE":   round(rf_rmse, 2),
            "RF_MAE":    round(rf_mae, 2),
            "LSTM_RMSE": round(lstm_rmse, 2),
            "LSTM_MAE":  round(lstm_mae, 2),
            "Winner":    "RF" if rf_rmse <= lstm_rmse else "LSTM"
        })

    # ── final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY — ALL DATASETS")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    # save summary
    summary_path = os.path.join(BASE_DIR, "outputs", "model_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Summary saved → {summary_path}")
    print("\n✅ All datasets processed successfully!")


if __name__ == "__main__":
    main()
