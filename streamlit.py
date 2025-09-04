import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, MaxPooling1D, Flatten
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import explained_variance_score, mean_poisson_deviance, mean_gamma_deviance, r2_score, max_error
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# --------------------------
# Streamlit App
# --------------------------
def main():
    st.title("Yahoo Finance Stock Prediction with CNN-LSTM")
    st.write("This app fetches stock data, trains a CNN-LSTM model, and predicts stock prices.")

    # --------------------------
    # User Inputs
    # --------------------------
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT):")
    period = st.selectbox("Select period:", ["1y", "2y", "5y", "10y", "max"])
    interval = st.selectbox("Select interval:", ["1d", "1wk", "1mo"])
    window_size = st.number_input("Sliding window size (timesteps):", min_value=5, max_value=60, value=20)

    if not ticker:
        st.info("Please enter a ticker symbol to continue.")
        return

    # --------------------------
    # Fetch Stock Data
    # --------------------------
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            st.error("No data found for this ticker.")
            return
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # --------------------------
    # Preprocess Data
    # --------------------------
    price_col = "Adj Close" if "Adj Close" in data.columns else data.columns[3]  # fallback to Close
    data['Price'] = data[price_col]
    st.subheader(f"Data Preview for {ticker}")
    st.dataframe(data[['Price']].tail(10))

    # Scale data
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(data['Price'].values.reshape(-1,1))

    # Create sequences for CNN-LSTM
    X, Y = [], []
    for i in range(window_size, len(scaled_prices)):
        X.append(scaled_prices[i-window_size:i, 0])
        Y.append(scaled_prices[i, 0])
    X, Y = np.array(X), np.array(Y)

    # Reshape for CNN-LSTM: (samples, timesteps, features, 1)
    X = X.reshape((X.shape[0], X.shape[1], 1, 1))
    Y = Y.reshape(-1,1)

    # Split into train/test
    split = int(0.8*len(X))
    train_X, test_X = X[:split], X[split:]
    train_Y, test_Y = Y[:split], Y[split:]

    st.write(f"Training samples: {len(train_X)}, Testing samples: {len(test_X)}")

    # --------------------------
    # Build CNN-LSTM Model
    # --------------------------
    model = tf.keras.Sequential()
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu'), input_shape=(None, 1, 1)))
    model.add(TimeDistributed(MaxPooling1D(1)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(100, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])

    st.subheader("Training CNN-LSTM Model...")
    history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y),
                        epochs=10, batch_size=32, verbose=1, shuffle=False)

    # --------------------------
    # Predictions and Metrics
    # --------------------------
    y_pred = model.predict(test_X).flatten()
    test_Y_flat = test_Y.flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    test_Y_inv = scaler.inverse_transform(test_Y_flat.reshape(-1,1)).flatten()

    st.subheader("Model Performance Metrics")
    st.write("R2 Score:", r2_score(test_Y_inv, y_pred_inv))
    st.write("Explained Variance Score:", explained_variance_score(test_Y_inv, y_pred_inv))
    st.write("Max Error:", max_error(test_Y_inv, y_pred_inv))
    st.write("Mean Poisson Deviance:", mean_poisson_deviance(test_Y_inv, y_pred_inv))
    st.write("Mean Gamma Deviance:", mean_gamma_deviance(test_Y_inv, y_pred_inv))

    # --------------------------
    # Plot Predictions
    # --------------------------
    st.subheader("Predicted vs Actual Prices")
    result_df = pd.DataFrame({'Actual': test_Y_inv, 'Predicted': y_pred_inv})
    st.line_chart(result_df)

if __name__ == "__main__":
    main()
