import pandas as pd
import streamlit as st
import yfinance as yf
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Function to preprocess data safely
def preprocess_data(df):
    # Detect the Adjusted Close column automatically
    adj_col_candidates = [col for col in df.columns if 'Adj' in col and 'Close' in col]
    if not adj_col_candidates:
        st.error("Adjusted Close column not found in the dataset!")
        return None
    adj_col = adj_col_candidates[0]
    
    df['Price'] = df[adj_col]
    # Additional preprocessing steps
    df['Date'] = pd.to_datetime(df.index) if df.index.name is None else pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.reset_index(drop=True, inplace=True)
    return df

# Streamlit App
def main():
    st.title("Stock Price Prediction App")

    st.write("Upload your stock CSV file or use yfinance ticker.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    ticker_input = st.text_input("Or enter a ticker symbol (e.g., AAPL)")

    if uploaded_file:
        stock_data = pd.read_csv(uploaded_file)
    elif ticker_input:
        stock_data = yf.download(ticker_input, period="1y")
    else:
        st.info("Please upload a CSV or enter a ticker symbol to continue.")
        return

    # Preprocess data
    stock_data = preprocess_data(stock_data)
    if stock_data is None:
        return

    st.subheader("Data Preview")
    st.dataframe(stock_data.head())

    # Additional code for model training, predictions, and visualization
    # For example: display line chart of prices
    st.subheader("Stock Price Over Time")
    st.line_chart(stock_data['Price'])

if __name__ == "__main__":
    main()
