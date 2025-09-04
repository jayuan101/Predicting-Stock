import streamlit as st
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# Streamlit App
def main():
    st.title("Yahoo Finance Stock Price Viewer")

    st.write("""
    Enter a stock ticker symbol (e.g., AAPL, MSFT) to fetch historical stock prices from Yahoo Finance.
    """)

    # User input for ticker
    ticker_input = st.text_input("Enter a ticker symbol:")

    # User input for time period
    period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])
    interval = st.selectbox("Select interval:", ["1d", "1wk", "1mo"])

    if ticker_input:
        try:
            # Download stock data from Yahoo Finance
            stock_data = yf.download(ticker_input, period=period, interval=interval)
            
            if stock_data.empty:
                st.error("No data found for this ticker.")
                return

            # Preprocess: create a Price column from Adjusted Close
            stock_data['Price'] = stock_data['Adj Close']

            st.subheader("Data Preview")
            st.dataframe(stock_data.head())

            st.subheader("Adjusted Close Price Over Time")
            st.line_chart(stock_data['Price'])

        except Exception as e:
            st.error(f"Error fetching data: {e}")

if __name__ == "__main__":
    main()
