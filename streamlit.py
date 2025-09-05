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
    r2_score,
    max_error,
)
from sklearn.ensemble import RandomForestRegressor

# --------------------------
# App Config
# --------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Helpers
# --------------------------
@st.cache_data(show_spinner=False)
def download_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return yf.download(ticker, period=period, interval=interval, auto_adjust=False)

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
    st.title("📈 Stock Price Prediction")
    st.markdown(
        """
        This app downloads stock price data from **Yahoo Finance**  
        and uses a **CNN-LSTM deep learning model** (or a fallback machine learning model)  
        to predict prices and compare them with actual values.  
        """
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        ticker = st.text_input("Stock Ticker", value="AAPL", help="Example: AAPL, MSFT, TSLA")
        period = st.selectbox("History Period", ["1y", "2y", "5y", "10y", "max"], index=0)
        interval = st.selectbox("Data Interval", ["1d", "1wk", "1mo"], index=0)
        window_size = st.slider("Window Size (timesteps)", 5, 120, 30, help="How many past days to use for prediction")
        test_size_pct = st.slider("Test Set (%)", 10, 40, 20, help="Portion of data reserved for testing")
        epochs = st.slider("Training Epochs (if TF available)", 5, 100, 20)
        batch_size = st.selectbox("Batch Size (if TF available)", [16, 32, 64, 128], index=2)
        seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)

    if not ticker:
        st.info("👆 Enter a stock ticker symbol in the sidebar to begin.")
        return

    # Download data
    try:
        with st.spinner("Fetching stock data..."):
            data = download_prices(ticker, period, interval)
        if data is None or data.empty:
            st.error("No data found for this ticker/period/interval.")
            return
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Select price column
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    data["Price"] = data[price_col]

    st.subheader(f"📊 Recent Data for {ticker}")
    st.dataframe(data.tail(10))

    # Scale and create sequences
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data["Price"].values.reshape(-1, 1))
    X, Y = make_sequences(scaled, window_size)
    X_tf = X.reshape((X.shape[0], X.shape[1], 1))  # for TF
    seq_dates = data.index[window_size:]

    # Train/test split
    test_size = max(1, int(len(X) * test_size_pct / 100.0))
    split = len(X) - test_size
    X_train_tf, X_test_tf = X_tf[:split], X_tf[split:]
    y_train, y_test = Y[:split], Y[split:]
    X_train_sk, X_test_sk = X[:split], X[split:]
    dates_test = seq_dates[split:]

    st.write(f"✅ Training samples: {len(X_train_tf)} | Testing samples: {len(X_test_tf)}")

    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)

    # Train model
    if TF_AVAILABLE:
        st.subheader("🤖 Training CNN-LSTM model...")
        with st.spinner("Training deep learning model..."):
            model = build_tf_model(window_size)
            callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
            model.fit(
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
        st.subheader("⚡ TensorFlow not available — using fallback model (RandomForest)")
        with st.spinner("Training RandomForest model..."):
            model_sk = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
            model_sk.fit(X_train_sk, y_train.ravel())
        y_pred_scaled = model_sk.predict(X_test_sk).reshape(-1, 1)

    # Inverse scaling
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    y_test_inv = scaler.inverse_transform(y_test).ravel()

    # Metrics
    st.subheader("📐 Model Performance (Test Set)")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2_score(y_test_inv, y_pred):.4f}")
    col2.metric("Explained Variance", f"{explained_variance_score(y_test_inv, y_pred):.4f}")
    col3.metric("Max Error", f"{max_error(y_test_inv, y_pred):.2f}")

    # Plot
    st.subheader("📉 Predicted vs Actual Prices")
    result_df = pd.DataFrame({"Actual": y_test_inv, "Predicted": y_pred}, index=pd.Index(dates_test, name="Date"))
    st.line_chart(result_df)

    # Download option
    with st.expander("⬇️ Download Results"):
        csv = result_df.to_csv(index=True).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name=f"{ticker}_pred_vs_actual.csv", mime="text/csv")


if __name__ == "__main__":
    main()
