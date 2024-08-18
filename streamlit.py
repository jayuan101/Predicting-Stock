import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, MaxPooling1D, Flatten

# Title and Description
st.title("Interactive Stock Data Viewer & Model Training")
st.write("Enter a stock ticker symbol to visualize the stock data and train a predictive model.")

# Input for stock ticker
ticker_symbol = st.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT)", "AAPL")

# Date range selection
start_date = st.date_input("Start date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End date", pd.to_datetime("today"))

# Fetching stock data
if ticker_symbol:
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    if not stock_data.empty:
        # Display the stock data
        st.write(f"### {ticker_symbol.upper()} Stock Data")
        st.dataframe(stock_data)

        # Preprocessing the data
        def preprocess_data(df):
            df['Date'] = df.index
            df['Price'] = df['Adj Close']
            df = df[['Date', 'Price']]
            df['Price'] = df['Price'].fillna(method='ffill')
            return df

        stock_data = preprocess_data(stock_data)
        data = stock_data['Price'].values.reshape(-1, 1)

        # Splitting the data
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]

        def create_dataset(data, time_step=100):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 100
        train_X, train_Y = create_dataset(train_data, time_step)
        test_X, test_Y = create_dataset(test_data, time_step)

        train_X = train_X.reshape(train_X.shape[0], 1, time_step, 1)
        test_X = test_X.reshape(test_X.shape[0], 1, time_step, 1)

        # Model Creation
        model = tf.keras.Sequential()
        model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, time_step, 1))))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(2)))
        model.add(TimeDistributed(Flatten()))

        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(100, return_sequences=False)))
        model.add(Dropout(0.5))

        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

        # Training the model
        if st.button("Train Model"):
            with st.spinner("Training the model..."):
                history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=40, batch_size=40, verbose=1, shuffle=True)
            st.success("Model training complete!")
            
            # Plot the training history
            st.write("### Training History")
            st.line_chart(history.history['mse'])
            st.line_chart(history.history['mae'])
    else:
        st.write("No data found for the given ticker symbol. Please try another symbol.")
else:
    st.write("Please enter a valid stock ticker symbol.")
