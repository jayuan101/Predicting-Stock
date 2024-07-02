pip install yfinance
import streamlit as st
import yfinance as yf
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Function to create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)].values
        y = data.iloc[i+seq_length].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Streamlit application
st.title("Stock Price Prediction with LSTM")

# Sidebar for stock ticker input
ticker = st.sidebar.text_input("Enter stock ticker", "AAPL")

# Fetch stock data using yfinance
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")
st.write(f"Displaying data for {ticker}")
st.line_chart(data["Close"])

# Prepare data for LSTM model
st.write("Preparing data for the model...")
data['Date'] = data.index
data.reset_index(drop=True, inplace=True)
data['Close'] = data['Close'].astype('float32')

# Create sequences for training
seq_length = 60
X, y = create_sequences(data[['Close']], seq_length)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
st.write("Defining the LSTM model...")
model = tf.keras.Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
st.write("Training the model...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Plot training history
st.write("Model Training History")
st.line_chart(history.history['loss'])
st.line_chart(history.history['val_loss'])

# Make predictions
st.write("Making predictions...")
predictions = model.predict(X_test)

# Plot predictions
st.write("Plotting predictions vs actual values")
st.line_chart(predictions.flatten(), label='Predictions')
st.line_chart(y_test.flatten(), label='Actual')

# Model evaluation metrics
st.write("Model Evaluation")
r2 = r2_score(y_test, predictions)
st.write(f"R2 Score: {r2:.2f}")

# Display model summary
st.write("Model Summary")
st.text(model.summary())
