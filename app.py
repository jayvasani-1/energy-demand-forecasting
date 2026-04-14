import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Demand Forecaster",
    page_icon="⚡",
    layout="wide"
)

# ── base paths ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
PREDS_DIR  = os.path.join(BASE_DIR, "outputs", "predictions")

# ── all 12 datasets ───────────────────────────────────────────────────────────
DATASETS = {
    "AEP":      {"file": "AEP_hourly.csv",      "target": "AEP_MW"},
    "COMED":    {"file": "COMED_hourly.csv",     "target": "COMED_MW"},
    "DAYTON":   {"file": "DAYTON_hourly.csv",    "target": "DAYTON_MW"},
    "DEOK":     {"file": "DEOK_hourly.csv",      "target": "DEOK_MW"},
    "DOM":      {"file": "DOM_hourly.csv",        "target": "DOM_MW"},
    "DUQ":      {"file": "DUQ_hourly.csv",        "target": "DUQ_MW"},
    "EKPC":     {"file": "EKPC_hourly.csv",       "target": "EKPC_MW"},
    "FE":       {"file": "FE_hourly.csv",         "target": "FE_MW"},
    "NI":       {"file": "NI_hourly.csv",         "target": "NI_MW"},
    "PJM Load": {"file": "PJM_Load_hourly.csv",   "target": "PJM_Load_MW"},
    "PJME":     {"file": "PJME_hourly.csv",       "target": "PJME_MW"},
    "PJMW":     {"file": "PJMW_hourly.csv",       "target": "PJMW_MW"},
}

# exact 6 features the model was trained on
FEATURE_COLS = ["hour", "day_of_week", "month", "lag_1", "lag_2", "lag_24"]

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file_path, target_col):
    df = pd.read_csv(file_path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = df[["Datetime", target_col]].dropna()
    
    # FIX 1: Limit data size on Streamlit Cloud to prevent memory crash
    MAX_ROWS = 30000
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS).reset_index(drop=True)
    return df
    
@st.cache_data
def engineer_features(df, target_col):
    """Matches Scripts/feature_engineering.py exactly — 6 features"""
    df = df.copy()
    # set Datetime as index (matches how feature_engineering.py uses data.index)
    df = df.set_index("Datetime")

    # time-based features
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month

    # lag features
    df["lag_1"]  = df[target_col].shift(1)
    df["lag_2"]  = df[target_col].shift(2)
    df["lag_24"] = df[target_col].shift(24)

    df = df.dropna()
    df = df.reset_index()  # bring Datetime back as column
    return df

@st.cache_resource
def load_saved_rf(dataset_key):
    path = os.path.join(MODELS_DIR, f"{dataset_key}_rf_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_saved_lstm(dataset_key):
    try:
        from tensorflow.keras.models import load_model
        path = os.path.join(MODELS_DIR, f"{dataset_key}_lstm_model.h5")
        if os.path.exists(path):
            return load_model(path)
    except Exception:
        pass
    return None

@st.cache_data
def load_saved_predictions(dataset_key, model_type):
    path = os.path.join(PREDS_DIR, f"{dataset_key}_{model_type}_forecast.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Datetime"])
    return None

@st.cache_data
def load_summary():
    path = os.path.join(BASE_DIR, "outputs", "model_comparison_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return rmse, mae, mape

def check_models_exist(dataset_key):
    rf_exists   = os.path.exists(os.path.join(MODELS_DIR, f"{dataset_key}_rf_model.pkl"))
    lstm_exists = os.path.exists(os.path.join(MODELS_DIR, f"{dataset_key}_lstm_model.h5"))
    return rf_exists, lstm_exists

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
    st.title("⚡ Controls")
    st.markdown("---")

    dataset_name = st.selectbox("Select Dataset", list(DATASETS.keys()))
    ds           = DATASETS[dataset_name]
    target_col   = ds["target"]
    file_path    = os.path.join(DATA_DIR, ds["file"])
    dataset_key  = ds["file"].split("_hourly")[0]

    model_choice = st.selectbox(
        "Select Model", ["Random Forest", "LSTM"],
        help="Random Forest is faster; LSTM captures long-term patterns"
    )

    rf_exists, lstm_exists = check_models_exist(dataset_key)
    st.markdown("### 🗂️ Pre-trained Models")
    st.write(f"RF:   {'✅ found' if rf_exists else '❌ not found'}")
    st.write(f"LSTM: {'✅ found' if lstm_exists else '❌ not found'}")

    st.markdown("### 📅 Date Range")
    if os.path.exists(file_path):
        df_raw     = load_data(file_path, target_col)
        min_date   = df_raw["Datetime"].min().date()
        max_date   = df_raw["Datetime"].max().date()
        start_date = st.date_input("From",
                                   value=max(min_date, pd.to_datetime("2016-01-01").date()),
                                   min_value=min_date, max_value=max_date)
        end_date   = st.date_input("To", value=max_date,
                                   min_value=min_date, max_value=max_date)
    else:
        st.error(f"File not found:\n{file_path}")
        st.stop()

    st.markdown("### ⚙️ Settings")
    test_split = st.slider("Test set size (%)", 10, 30, 20)
    show_shap  = st.checkbox("Show SHAP values", value=True)

    st.markdown("---")
    st.caption("MSc Thesis · Jay Vasani")
    st.caption("University of Europe for Applied Sciences · 2025")

# ── main header ───────────────────────────────────────────────────────────────
st.title("⚡ Short-Term Energy Demand Forecasting")
st.caption(f"LSTM vs Random Forest  ·  Dataset: {dataset_name} ({target_col})  ·  MSc Thesis Project")

# ── load + filter ─────────────────────────────────────────────────────────────
df      = engineer_features(df_raw, target_col)
mask    = (df["Datetime"].dt.date >= start_date) & (df["Datetime"].dt.date <= end_date)
df_filtered = df[mask].copy()

if len(df_filtered) < 200:
    st.warning("Please select a wider date range (at least a few months).")
    st.stop()

# FIX 4: Keep as DataFrame (remove .values) to fix UserWarning about feature names
X               = df_filtered[FEATURE_COLS]
y               = df_filtered[target_col].values
split           = int(len(X) * (1 - test_split / 100))

# Use .iloc for DataFrame slicing
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]
dates_test      = df_filtered["Datetime"].values[split:]

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Forecast", "📊 Metrics", "⚔️ Model Comparison",
    "🏆 All Datasets", "🔍 Feature Importance", "📂 Raw Data"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if model_choice == "Random Forest":
        saved_model = load_saved_rf(dataset_key)
        if saved_model:
            st.success("✅ Loaded pre-trained Random Forest model from disk.")
            model  = saved_model
            y_pred = model.predict(X_test)
        else:
            with st.spinner("No saved model — training Random Forest live..."):
                from sklearn.ensemble import RandomForestRegressor
                model  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            st.warning("Trained live. Run main.py to save models for instant loading.")

    else:  # LSTM
        saved_lstm = load_saved_lstm(dataset_key)
        if saved_lstm:
            st.success("✅ Loaded pre-trained LSTM model from disk.")
            from sklearn.preprocessing import MinMaxScaler
            scaler_X  = MinMaxScaler()
            scaler_y  = MinMaxScaler()
            scaler_X.fit(X.iloc[:split]) # Fit on train portion
            X_test_s  = scaler_X.transform(X_test)
            scaler_y.fit(y_train.reshape(-1, 1))
            X_test_r  = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
            y_pred    = scaler_y.inverse_transform(saved_lstm.predict(X_test_r)).flatten()
            model     = None
        else:
            with st.spinner("No saved LSTM — training live (few minutes)..."):
                try:
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout
                    from sklearn.preprocessing import MinMaxScaler
                    scaler_X  = MinMaxScaler()
                    scaler_y  = MinMaxScaler()
                    scaler_X.fit(X.iloc[:split])
                    X_train_s = scaler_X.transform(X_train)
                    X_test_s  = scaler_X.transform(X_test)
                    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1))
                    X_train_r = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
                    X_test_r  = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
                    lstm_m = Sequential([
                        LSTM(64, input_shape=(1, X_train_s.shape[1]), return_sequences=True),
                        Dropout(0.2),
                        LSTM(32),
                        Dropout(0.2),
                        Dense(1)
                    ])
                    lstm_m.compile(optimizer="adam", loss="mse")
                    lstm_m.fit(X_train_r, y_train_s, epochs=5, batch_size=64, verbose=0)
                    y_pred = scaler_y.inverse_transform(lstm_m.predict(X_test_r)).flatten()
                    model  = None
                except Exception as e:
                    st.error(f"LSTM failed: {e}")
                    st.stop()

    rmse, mae, mape = compute_metrics(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{rmse:,.1f} MW")
    m2.metric("MAE",  f"{mae:,.1f} MW")
    m3.metric("MAPE", f"{mape:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_test, y=y_test,
                             name="Actual", line=dict(color="#2196F3", width=1.5)))
    fig.add_trace(go.Scatter(x=dates_test, y=y_pred,
                             name="Predicted", line=dict(color="#FF5722", width=1.5, dash="dot")))
    fig.update_layout(
        title=f"{model_choice}  —  {dataset_name}: Actual vs Predicted",
        xaxis_title="Date", yaxis_title=f"Energy Demand ({target_col})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified", height=440,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    # FIX 3: Replaced use_container_width with width='stretch'
    st.plotly_chart(fig, width='stretch')

    pred_df = pd.DataFrame({
        "Datetime": dates_test, "Actual_MW": y_test, "Predicted_MW": y_pred
    })
    st.download_button("⬇️ Download predictions CSV",
                       pred_df.to_csv(index=False).encode(),
                       f"{dataset_key}_{model_choice.lower().replace(' ','_')}_predictions.csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:,.1f} MW",  help="Root Mean Squared Error — lower is better")
    c2.metric("MAE",  f"{mae:,.1f} MW",   help="Mean Absolute Error — lower is better")
    c3.metric("MAPE", f"{mape:.2f}%",     help="Mean Absolute Percentage Error")

    st.markdown("---")
    residuals = y_test - y_pred
    col1, col2 = st.columns(2)

    with col1:
        fig_res = px.histogram(residuals, nbins=60,
                               title="Residual Distribution",
                               labels={"value": "Residual (MW)"},
                               color_discrete_sequence=["#2196F3"])
        fig_res.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_res, width='stretch')

    with col2:
        fig_sc = px.scatter(x=y_test, y=y_pred, opacity=0.3,
                            title="Actual vs Predicted (scatter)",
                            labels={"x": "Actual MW", "y": "Predicted MW"},
                            color_discrete_sequence=["#FF5722"])
        mn, mx = float(y_test.min()), float(y_test.max())
        fig_sc.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                         line=dict(color="gray", dash="dash"))
        fig_sc.update_layout(height=320)
        st.plotly_chart(fig_sc, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RF vs LSTM COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"RF vs LSTM — {dataset_name}")
    rf_preds   = load_saved_predictions(dataset_key, "rf")
    lstm_preds = load_saved_predictions(dataset_key, "lstm")

    if rf_preds is not None and lstm_preds is not None:
        merged = pd.merge(
            rf_preds.rename(columns={"Forecast_MW": "RF_MW"}),
            lstm_preds.rename(columns={"Forecast_MW": "LSTM_MW"}),
            on="Datetime", how="inner"
        )
        if len(merged) > 5000:
            merged = merged.iloc[::4].reset_index(drop=True)

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=merged["Datetime"], y=merged["RF_MW"],
                                     name="Random Forest", line=dict(color="#4CAF50", width=1.2)))
        fig_cmp.add_trace(go.Scatter(x=merged["Datetime"], y=merged["LSTM_MW"],
                                     name="LSTM", line=dict(color="#9C27B0", width=1.2)))
        fig_cmp.update_layout(
            title=f"{dataset_name} — RF vs LSTM Full Forecast",
            xaxis_title="Date", yaxis_title="MW",
            hovermode="x unified", height=420,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cmp, width='stretch')
    else:
        st.info("Run main.py first to generate saved predictions for both models.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALL DATASETS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🏆 All Datasets — RF vs LSTM Summary")
    summary = load_summary()
    if summary is not None:
        st.dataframe(
            summary.style.map(
                lambda v: "background-color: #e8f5e9; color: #2e7d32;" if v == "RF"
                else "background-color: #f3e5f5; color: #6a1b9a;" if v == "LSTM" else "",
                subset=["Winner"]
            ),
            width='stretch'
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fig_rmse = px.bar(
                summary, x="Dataset", y=["RF_RMSE", "LSTM_RMSE"],
                barmode="group", title="RMSE Comparison — All Datasets",
                color_discrete_map={"RF_RMSE": "#4CAF50", "LSTM_RMSE": "#9C27B0"}
            )
            fig_rmse.update_layout(height=380, xaxis_tickangle=-45)
            st.plotly_chart(fig_rmse, width='stretch')

        with col2:
            fig_mae = px.bar(
                summary, x="Dataset", y=["RF_MAE", "LSTM_MAE"],
                barmode="group", title="MAE Comparison — All Datasets",
                color_discrete_map={"RF_MAE": "#4CAF50", "LSTM_MAE": "#9C27B0"}
            )
            fig_mae.update_layout(height=380, xaxis_tickangle=-45)
            st.plotly_chart(fig_mae, width='stretch')

        rf_wins   = (summary["Winner"] == "RF").sum()
        lstm_wins = (summary["Winner"] == "LSTM").sum()
        st.markdown(f"**Overall: Random Forest wins {rf_wins}/12 datasets, "
                    f"LSTM wins {lstm_wins}/12 datasets.**")
    else:
        st.info("Run main.py to generate the summary CSV.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Feature Importance")
    if model_choice == "Random Forest" and model is not None:
        importances = pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(importances, x="Importance", y="Feature",
                        orientation="h", title="Random Forest Feature Importances",
                        color="Importance", color_continuous_scale="Blues")
        fig_fi.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_fi, width='stretch')

        if show_shap:
            try:
                import shap
                with st.spinner("Computing SHAP values..."):
                    explainer   = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test.iloc[:300])
                shap_df = pd.DataFrame(
                    np.abs(shap_values).mean(axis=0),
                    index=FEATURE_COLS, columns=["SHAP"]
                ).sort_values("SHAP", ascending=True)

                fig_shap = px.bar(shap_df.reset_index(), x="SHAP", y="index",
                                  orientation="h", title="SHAP Feature Importance",
                                  color="SHAP", color_continuous_scale="Oranges")
                fig_shap.update_layout(height=350, showlegend=False, yaxis_title="Feature")
                st.plotly_chart(fig_shap, width='stretch')
            except Exception:
                st.info("SHAP calculation skipped (usually due to memory limits).")
    else:
        st.info("Select Random Forest in the sidebar to see feature importance.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader(f"Dataset Preview — {dataset_name}")
    show_cols = ["Datetime", target_col] + FEATURE_COLS
    st.dataframe(df_filtered[show_cols].head(500), width='stretch')
    st.caption(f"Showing first 500 of {len(df_filtered):,} rows in selected date range.")

    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered dataset",
                       csv, f"{dataset_key}_filtered.csv", "text/csv")
