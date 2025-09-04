# Stock Price Prediction App

This is a **Streamlit-based web application** for exploring and predicting stock prices. The app allows users to upload CSV stock data or fetch stock data directly from Yahoo Finance using ticker symbols. It includes preprocessing, visualization, and the ability to integrate machine learning models for prediction.

---

## Features

- Upload a CSV file containing stock price data.
- Fetch stock data using a ticker symbol (e.g., AAPL, MSFT) via `yfinance`.
- Automatic detection of the **Adjusted Close** column for price analysis.
- Interactive data preview using Streamlit’s table and chart components.
- Visualization of stock prices over time with interactive line charts.
- Preprocessing ensures data is sorted by date and ready for modeling.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/jayuan101/Predicting-Stock.git
cd Predicting-Stock
