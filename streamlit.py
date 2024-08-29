import streamlit as st
import yfinance as yf
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import plot_importance
import requests

# News API Key (replace with your own)
NEWS_API_KEY = st.secrets['API_key']

# Title and Description
st.title("Interactive Stock Data Viewer, Model Training & News Feed")
st.write("Enter a stock ticker symbol to visualize the stock data, train an XGBoost model, and view related news.")

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
        data = stock_data['Price'].values

        def create_lagged_features(data, n_lags=5):
            df = pd.DataFrame(data)
            for i in range(1, n_lags + 1):
                df[f'lag_{i}'] = df[0].shift(i)
            df.dropna(inplace=True)
            return df

        n_lags = 5
        df_lagged = create_lagged_features(data, n_lags)

        X = df_lagged.iloc[:, 1:].values  # Features: lagged prices
        y = df_lagged.iloc[:, 0].values  # Target: current price

        # Splitting the data
        train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Creation with XGBoost
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)

        # Button to train the model
        if st.button("Train Model"):
            with st.spinner("Training the XGBoost model..."):
                # Train the model on the training data
                model.fit(train_X, train_Y)
            st.success("Model training complete!")

            # Make predictions
            predictions = model.predict(test_X)

            # Model performance metrics
            mse = mean_squared_error(test_Y, predictions)
            mae = mean_absolute_error(test_Y, predictions)

            # Display the results
            st.write(f"### Model Performance")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

            # Plotting the feature importance using Matplotlib
            st.write("### Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_importance(model, ax=ax)
            st.pyplot(fig)

            # Correlation heatmap (optional)
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            corr = pd.DataFrame(train_X, columns=[f'lag_{i}' for i in range(1, n_lags + 1)]).corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# Fetching and displaying news
news_url = f"https://newsapi.org/v2/everything?q={ticker_symbol}&apiKey={NEWS_API_KEY}"
news_response = requests.get(news_url)
news_data = news_response.json()

if news_data.get("status") == "ok":
    articles = news_data.get("articles", [])
    if articles:
        st.write(f"### Latest News for {ticker_symbol.upper()}")
        for article in articles[:5]:  # Limit to 5 news articles
            st.write(f"#### {article['title']}")
            st.write(article['description'])
            st.write(f"[Read more]({article['url']})")
    else:
        st.write("No news articles found.")
else:
    st.write("Failed to retrieve news.")
