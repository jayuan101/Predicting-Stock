# 📈 Stock Price Prediction with CNN-LSTM (Streamlit App)

This is an interactive **Streamlit web app** that allows you to:

- Fetch historical stock data directly from **Yahoo Finance**
- Train a **CNN-LSTM deep learning model** on stock prices
- Predict and visualize future price movements
- View model performance with key evaluation metrics

---

## 🚀 Features
- **Ticker Input**: Enter any stock ticker (e.g., `AAPL`, `MSFT`, `TSLA`)
- **Customizable Options**:
  - Select data period (`1y`, `2y`, `5y`, `10y`, or `max`)
  - Choose data interval (`1d`, `1wk`, or `1mo`)
  - Adjust the sliding window size for training
- **Model Training**:
  - Hybrid **CNN + LSTM** neural network
  - Train/test split with real-time training feedback
- **Performance Metrics**:
  - R² Score
  - Explained Variance
  - Max Error
  - Mean Poisson Deviance
  - Mean Gamma Deviance
- **Visualizations**:
  - Interactive preview of stock data
  - Line chart comparing actual vs predicted prices

---

## 🛠 Installation

### 1. Clone this repository
```bash
git clone https://github.com/your-username/stock-cnn-lstm-streamlit.git
cd stock-cnn-lstm-streamlit

2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt


requirements.txt should contain:

streamlit
pandas
numpy
yfinance
tensorflow
scikit-learn

▶️ Run the App
streamlit run app.py


Then open your browser at http://localhost:8501

📊 Example

Enter ticker: AAPL

Period: 2y

Interval: 1d

Window size: 20

Train the model and view predictions 📉📈


├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


🧠 Model Architecture

CNN Layers for feature extraction

Bidirectional LSTMs for sequence learning

Dropout layers for regularization

Dense layer for final prediction

⚠️ Disclaimer

This project is for educational purposes only.
It is not financial advice — use predictions responsibly.

🙌 Acknowledgements

Streamlit

Yahoo Finance API (yfinance)

TensorFlow / Keras
