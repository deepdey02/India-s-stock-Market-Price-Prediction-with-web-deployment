from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import ta

app = Flask(__name__)

# Load model and scaler once
model = load_model('LSTM_Model.h5')
scaler = joblib.load('scaler.pkl')
features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input prices
        prices_str = request.form['prices']
        prices = [float(p.strip()) for p in prices_str.split(',')]

        if len(prices) != 60:
            return render_template('result.html', prediction="❌ Please enter exactly 60 prices.")

        # Load last 100 close prices
        tail_df = pd.read_csv("reliance_tail_100.csv")
        combined_df = pd.concat([tail_df, pd.DataFrame({'Close': prices})], ignore_index=True)

        # Calculate indicators
        combined_df['SMA_20'] = ta.trend.sma_indicator(combined_df['Close'], window=20)
        combined_df['SMA_50'] = ta.trend.sma_indicator(combined_df['Close'], window=50)
        combined_df['RSI'] = ta.momentum.rsi(combined_df['Close'], window=14)
        combined_df['MACD'] = ta.trend.macd_diff(combined_df['Close'], window_slow=26, window_fast=12, window_sign=9)

        # Get last 60 rows
        latest_input = combined_df[features].tail(60).values
        scaled_input = scaler.transform(latest_input)
        X = scaled_input.reshape(1, 60, len(features))

        # Predict
        pred_scaled = model.predict(X)
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = pred_scaled[0, 0]
        pred_price = scaler.inverse_transform(dummy)[0, 0]

        return render_template('result.html', prediction=f"🟢 Predicted Next Closing Price: ₹{pred_price:.2f}")

    except Exception as e:
        return render_template('result.html', prediction=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
