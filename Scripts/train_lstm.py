import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_lstm(X, y, sequence_length=24, epochs=50, batch_size=32):
    """
    Train an LSTM model for energy demand forecasting.
    
    Parameters:
    - X: Input features as a NumPy array.
    - y: Target variable as a NumPy array.
    - sequence_length: Number of time steps for LSTM input.
    - epochs: Number of training epochs.
    - batch_size: Size of each training batch.
    
    Returns:
    - model: Trained LSTM model.
    - X_test_seq: Sequence data for testing.
    - y_test: Corresponding target values for testing.
    """
    # Scale data to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    
    # Create sequences
    def create_sequences(features, target, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(features) - seq_length):
            X_seq.append(features[i:i + seq_length])
            y_seq.append(target[i + seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    
    # Train-test split
    X_train_seq, X_test_seq, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    # Build LSTM model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(sequence_length, X.shape[1]), return_sequences=True),
        LSTM(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test), epochs=epochs,
              batch_size=batch_size, callbacks=[early_stopping], verbose=1)
    
    return model, X_test_seq, y_test, scaler
