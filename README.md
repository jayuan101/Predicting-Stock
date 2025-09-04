# Yahoo Finance Stock Prediction App with CNN-LSTM

This Streamlit web app allows users to fetch historical stock data from **Yahoo Finance**, train a **CNN-LSTM neural network**, and predict future stock prices. It also provides interactive charts and multiple regression evaluation metrics.

---

## Features

- Fetch stock data directly from **Yahoo Finance** using ticker symbols (e.g., `AAPL`, `MSFT`).  
- Automatically handles **Adjusted Close** or fallback price columns.  
- Preprocesses data with **sliding windows** and scales it for neural network training.  
- Trains a **CNN-LSTM** model for stock price prediction.  
- Calculates and displays advanced regression metrics:
  - R² Score  
  - Explained Variance Score  
  - Max Error  
  - Mean Poisson Deviance  
  - Mean Gamma Deviance  
- Interactive line chart comparing **predicted vs actual prices**.  

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/jayuan101/Predicting-Stock.git
cd Predicting-Stock
