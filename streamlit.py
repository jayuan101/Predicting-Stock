import streamlit as st
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

def main():
    st.title("Yahoo Finance Stock Price Viewer")
    st.write("Enter a stock ticker symbol (e.g., AAPL, MSFT) to fetch historical stock prices from Yahoo Finance.")

    ticker_input = st.text_input("Enter a ticker symbol:")
    period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])
    interval = st.selectbox("Select interval:", ["1d", "1wk", "1mo"])

    if ticker_input:
        try:
            stock_data = yf.download(ticker_input, period=period, interval=interval)
            
            if stock_data.empty:
                st.error("No data found for this ticker.")
                return

            # Robust handling of price columns
            price_col = None
            for col in ['Adj Close', 'Close', 'Adj Open', 'Open']:
                if col in stock_data.columns:
                    price_col = col
                    break
            
            if price_col is None:
                st.error("No recognizable price column found in the data.")
                return

            stock_data['Price'] = stock_data[price_col]

            st.subheader(f"Data Preview ({ticker_input})")
            st.dataframe(stock_data.head())

            st.subheader(f"Price Over Time ({price_col})")
            st.line_chart(stock_data['Price'])

        except Exception as e:
            st.error(f"Error fetching data: {e}")

if __name__ == "__main__":
    main()
