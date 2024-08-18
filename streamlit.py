import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Title and Description
st.title("Interactive Stock Data Viewer")
st.write("Enter a stock ticker symbol to visualize the stock data.")

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

        # Select Columns to Display
        columns = st.multiselect("Select columns to plot", stock_data.columns.tolist(), default=['Adj Close'])
        
        # Plotting
        st.write("### Stock Price Over Time")
        plt.figure(figsize=(10, 5))
        for col in columns:
            plt.plot(stock_data.index, stock_data[col], label=col)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"{ticker_symbol.upper()} Price Over Time")
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("No data found for the given ticker symbol. Please try another symbol.")
else:
    st.write("Please enter a valid stock ticker symbol.")
