import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TensorFlow log noise

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Try TensorFlow; if unavailable, fallback to scikit-learn
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    explained_variance_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
    r2_score,
    max_error,
)
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Stock Prediction (CNN-LSTM)", layout="wide")


# --------------------------
# Helpers
# --------------------------
@st.cache_data(show_spinner=False)
def download_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    return df

def make_sequences(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i, 0])
        y.append(series[i, 0])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y

def build_tf_model(window_size: int) -> "tf.keras.Model":
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


# --------------------------
# Streamlit App
# --------------------------
def main():
    st.title("Yahoo Finance Stock Prediction — CNN-LSTM (with fallback)")
    st.write("Fetch historical stock data, train a CNN-LSTM model, and compare predictions with actual prices.")

    with st.sidebar:
        st.subheader("Parameters")
        ticker = st.text_input("Ticker", value="AAPL", help="Example: AAPL, MSFT, TSLA")
        period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=0)
        interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
        window_size = st.number_input("Sliding window size (timesteps)", min_value=5, max_value=120, value=30, step=1)
        test_size_pct = st.slider("Test set percentage", 10, 40, 20, help="Fraction of sequences for testing")
        epochs = st.number_input("Epochs (TensorFlow)", min_value=1, max_value=200, value=20)
        batch_size = st.number_input("Batch size (TensorFlow)", min_value=8, max_value=512, value=64, step=8)
        seed = st.number_input("Random seed", min_value=0, max_value=2**32-1, value=42, step=1)

    if not ticker:
        st.info("Enter a stock ticker to continue.")
        return

    # Download data
    try:
        with st.spinner("Downloading data from Yahoo Finance..."):
            data = download_prices(ticker, period, interval)
        if data is None or data.empty:
            st.error("No data found for this ticker/period/interval.")
            return
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Select price column
    price_col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else data.columns[3])
    data = data.copy()
    data["Price"] = data[price_col]
    st.subheader(f"Preview of {ticker}")
    st.dataframe(data.tail(10))

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data["Price"].values.reshape(-1, 1))

    # Create sequences
    X, Y = make_sequences(scaled, window_size)
    if len(X) < 50:
        st.warning("Too few samples after creating sequences. Try a larger period or smaller window.")
    X_tf = X.reshape((X.shape[0], X.shape[1], 1))  # for TF
    seq_dates = data.index[window_size:]           # align dates with labels

    # Train/test split
    test_size = int(len(X) * test_size_pct / 100.0)
    test_size = max(test_size, 1)
    split = len(X) - test_size

    X_train_tf, X_test_tf = X_tf[:split], X_tf[split:]
    y_train, y_test = Y[:split], Y[split:]

    X_train_sk, X_test_sk = X[:split], X[split:]
    dates_test = seq_dates[split:]

    st.write(f"**Train samples:** {len(X_train_tf)} | **Test samples:** {len(X_test_tf)}")

    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)

    # Train model
    use_fallback = False
    if TF_AVAILABLE:
        st.subheader("Training CNN-LSTM model (TensorFlow)...")
        with st.spinner("Training..."):
            model = build_tf_model(window_size)
            callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
            history = model.fit(
                X_train_tf, y_train,
                validation_data=(X_test_tf, y_test),
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=0,
                shuffle=False,
                callbacks=callbacks,
            )
        y_pred_scaled = model.predict(X_test_tf, verbose=0).reshape(-1, 1)
    else:
        use_fallback = True

    if use_fallback:
        st.subheader("TensorFlow not available. Using fallback model (RandomForest).")
        model_sk = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
        with st.spinner("Training RandomForest..."):
            model_sk.fit(X_train_sk, y_train.ravel())
        y_pred_scaled = model_sk.predict(X_test_sk).reshape(-1, 1)

    # Inverse scaling
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    y_test_inv = scaler.inverse_transform(y_test).ravel()

    # Metrics
    st.subheader("Performance Metrics (Test set)")
    try:
        mpd = mean_poisson_deviance(y_test_inv, y_pred)
    except Exception:
        mpd = np.nan
    try:
        mgd = mean_gamma_deviance(y_test_inv, y_pred)
    except Exception:
        mgd = np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{r2_score(y_test_inv, y_pred):.4f}")
    col2.metric("Explained Var", f"{explained_variance_score(y_test_inv, y_pred):.4f}")
    col3.metric("Max Error", f"{max_error(y_test_inv, y_pred):.2f}")
    st.caption(f"Mean Poisson Deviance: {mpd:.4f} | Mean Gamma Deviance: {mgd:.4f}")

    # Chart
    st.subheader("Predicted vs Actual Prices")
    result_df = pd.DataFrame({"Actual": y_test_inv, "Predicted": y_pred}, index=pd.Index(dates_test, name="Date"))
    st.line_chart(result_df)

    with st.expander("Download Results (CSV)"):
        csv = result_df.to_csv(index=True).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name=f"{ticker}_pred_vs_actual.csv", mime="text/csv")

    if use_fallback:
        st.info(
            "Running in **fallback mode** (scikit-learn) because TensorFlow is not available.\n\n"
            "On Streamlit Community you can try adding `tensorflow-cpu` to `requirements.txt`. "
            "If the build fails, keep using fallback mode."
        )


if __name__ == "__main__":
    main()
