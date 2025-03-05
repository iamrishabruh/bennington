import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from utils.data_access import load_historical_data

def remove_outliers_quantile(df, col="Close", lower_q=0.01, upper_q=0.99):
    """
    Removes rows in df where df[col] is outside the [lower_q, upper_q] quantile range.
    Helps eliminate extreme dips/spikes from invalid data.
    """
    low_val = df[col].quantile(lower_q)
    high_val = df[col].quantile(upper_q)
    return df[df[col].between(low_val, high_val)]

def add_RSI(df, window=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def create_dataset(feature_df, target_series, look_back=60):
    X, y = [], []
    for i in range(len(feature_df) - look_back):
        window_features = feature_df.iloc[i : i+look_back].values
        target_value = target_series.iloc[i + look_back]
        X.append(window_features)
        y.append(target_value)
    return np.array(X), np.array(y)

def train_lstm_model(data_path, look_back=60, epochs=10, batch_size=32):
    """
    1. Loads historical data from CSV
    2. Removes outliers from 'Close'
    3. Computes RSI & SMA if missing
    4. Separately scales 'Close' vs. other features
    5. Creates train/test sets
    6. Trains the LSTM
    7. Plots Actual vs. Predicted on the correct (unscaled) scale
       with distinct markers/lines so the predicted line is clearly visible.
    """
    # 1. Load data
    df = load_historical_data(data_path).fillna(method="ffill")
    if df.empty:
        print("No data loaded. Exiting.")
        return None, None

    # 2. Remove outliers from 'Close'
    df = remove_outliers_quantile(df, col="Close", lower_q=0.01, upper_q=0.99)
    df = df.fillna(method="ffill")
    if df.empty:
        print("All data removed by outlier filtering. Exiting.")
        return None, None

    # 3. Compute RSI & SMA if missing
    if "SMA_20" not in df.columns:
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
    if "RSI" not in df.columns:
        df = add_RSI(df)

    # Convert columns to numeric if needed
    for col in ["Volume", "News_Sentiment", "SMA_20", "RSI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(method="ffill")

    # Drop any remaining NaNs
    df.dropna(subset=["Close", "Volume", "SMA_20", "RSI", "News_Sentiment"], inplace=True)
    if len(df) < 2 * look_back:
        print("Not enough data after outlier removal to form a train/test set.")
        return None, None

    # 4. Separate scalers for 'Close' vs. other features
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    other_scaler = MinMaxScaler(feature_range=(0, 1))

    close_data = df[["Close"]].values
    close_data_scaled = close_scaler.fit_transform(close_data)

    other_features = df[["Volume", "SMA_20", "RSI", "News_Sentiment"]].values
    other_data_scaled = other_scaler.fit_transform(other_features)

    combined_scaled = np.hstack([close_data_scaled, other_data_scaled])
    scaled_df = pd.DataFrame(combined_scaled, index=df.index,
                             columns=["CloseScaled", "VolScaled", "SMA_20_Scaled", "RSI_Scaled", "NewsSent_Scaled"])

    # 5. Create dataset
    close_series_scaled = pd.Series(close_data_scaled.ravel(), index=df.index)
    X, y = create_dataset(scaled_df, close_series_scaled, look_back=look_back)
    X = X.reshape(X.shape[0], X.shape[1], scaled_df.shape[1])

    # 6. Split train/test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    if len(X_test) == 0:
        print("No test data after split. Possibly not enough data.")
        return None, None

    # 7. Build & train LSTM
    model = tf.keras.models.Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, scaled_df.shape[1])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # 8. Predict on test
    preds_scaled = model.predict(X_test)
    preds_unscaled = close_scaler.inverse_transform(preds_scaled)
    actual_unscaled = close_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Align array lengths
    min_len = min(len(preds_unscaled), len(actual_unscaled))
    preds_unscaled = preds_unscaled[:min_len]
    actual_unscaled = actual_unscaled[:min_len]

    if min_len == 0:
        print("No overlapping predictions. Check your data or look_back settings.")
        return model, (close_scaler, other_scaler)

    # 9. Plot Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_len), actual_unscaled, color="blue", label="Actual", linewidth=2)
    # Use markers for predicted so it's visible even if it's close to actual
    plt.plot(range(min_len), preds_unscaled, color="red", label="Predicted", linewidth=2,
             marker="o", markersize=3)
    plt.title("LSTM Prediction (Unscaled Close Price)")
    plt.xlabel("Time Steps")
    plt.ylabel("Close Price")
    plt.legend()

    # Show in Streamlit or fallback
    try:
        import streamlit as st
        st.pyplot(plt.gcf())
    except ImportError:
        plt.show()

    return model, (close_scaler, other_scaler)
def adjust_strategy_with_predictions(strategy, model, scalers, data, look_back=60):
    close_scaler, other_scaler = scalers
    
    if "SMA_20" not in data.columns:
        data["SMA_20"] = data["Close"].rolling(window=20).mean().fillna(method="ffill")
    if "RSI" not in data.columns:
        data = add_RSI(data)
    
    for col in ["Volume", "News_Sentiment", "SMA_20", "RSI"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(method="ffill")
    
    recent_data = data.iloc[-look_back:].copy()
    close_vals = recent_data[["Close"]].values
    close_scaled = close_scaler.transform(close_vals)
    other_vals = recent_data[["Volume", "SMA_20", "RSI", "News_Sentiment"]].values
    other_scaled = other_scaler.transform(other_vals)
    combined_scaled = np.hstack([close_scaled, other_scaled])
    X_recent = combined_scaled.reshape(1, look_back, combined_scaled.shape[1])
    
    pred_scaled = model.predict(X_recent)[0, 0]
    predicted_close = close_scaler.inverse_transform([[pred_scaled]])[0, 0]
    last_close = recent_data["Close"].iloc[-1]
    
    # Adjust entry threshold
    if predicted_close > last_close:
        strategy.entry_threshold *= 0.95
    else:
        strategy.entry_threshold *= 1.05
    
    # Adjust base trade size based on volatility.
    returns = recent_data["Close"].pct_change().dropna()
    vol = returns.std() if not returns.empty else 0.02
    target_vol = 0.02
    strategy.base_trade_size = int(strategy.base_trade_size * (target_vol / vol))
    
    return strategy