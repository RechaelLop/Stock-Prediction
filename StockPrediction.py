# ===========================
# Stock Price Prediction App
# ===========================

# Import libraries
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import streamlit as st
import yfinance as yf

# ---------------------------
# Constants
# ---------------------------
TODAY = dt.date.today()  # current date

# ===========================
# Function: Fetch historical data
# ===========================
@st.cache_data
def get_stock_data(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    End date is clamped to today if necessary.
    Returns a DataFrame or None if no data is found.
    """
    if start >= end:
        st.error("❌ Start date must be before end date.")
        return None
    if end > TODAY:
        end = TODAY  # cannot fetch future data

    try:
        data = yf.download(symbol, start=start, end=end + dt.timedelta(days=1), progress=False)
        if data.empty:
            st.error("⚠️ No data available for this symbol/date range.")
            return None
        return data
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        return None

# ===========================
# Function: View closing price for a specific date
# ===========================
def view_close_price(symbol: str):
    """Displays the closing price for a selected historical date."""
    date = st.sidebar.date_input("Enter date to view close price (historical):")
    df = yf.download(symbol, start=date, end=date + dt.timedelta(days=1), progress=False)

    if df.empty:
        st.warning("⚠️ No data available for this date.")
    else:
        # Safely get first row
        close_price = df['Close'].iloc[0]
        st.write(f"Close price of {symbol} on {date}: ${close_price:.2f}")

# ===========================
# Function: Train LSTM and predict future price
# ===========================
def predict_stock(symbol: str, start: dt.date, end: dt.date, future_date: dt.date):
    """
    Shows historical data and predicts future stock price using LSTM.
    Handles any historical range, and predicts for any future date.
    """
    df = get_stock_data(symbol, start, end)
    if df is None:
        return

    # ---------------------------
    # Historical data visualization
    # ---------------------------
    st.subheader("📊 Historical Stock Data")
    st.dataframe(df)
    st.subheader("📈 Closing Price History")
    st.line_chart(df["Close"])

    # ---------------------------
    # Prepare data for LSTM
    # ---------------------------
    if "Close" not in df.columns:
        st.error("⚠️ 'Close' column not found in fetched data. Check stock symbol.")
        return

    data = df[["Close"]]
    dataset = data.values

    lookback = min(60, len(dataset))  # last 60 days or less if insufficient data
    scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataset)

    # Create sequences for LSTM training
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i - lookback:i, 0])
        y_train.append(scaled_data[i, 0])

    # Handle very short datasets
    if len(x_train) == 0:
        x_train.append(scaled_data[-lookback:])
        y_train.append(scaled_data[-1, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # ---------------------------
    # Build LSTM model
    # ---------------------------
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train model
    with st.spinner("⏳ Training model..."):
        model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    # ---------------------------
    # Predict future price
    # ---------------------------
    st.subheader(f"🔮 Predicted Close Price for {future_date}")

    # Use last 'lookback' days from historical data for prediction
    last_days = dataset[-lookback:]
    scaler_last = MinMaxScaler(feature_range=(0, 1))
    last_scaled = scaler_last.fit_transform(last_days)

    X_future = np.array([last_scaled])
    X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
    pred_scaled = model.predict(X_future)
    pred_price = scaler_last.inverse_transform(pred_scaled)

    st.write(f"${pred_price[0][0]:.2f}")

# ===========================
# Streamlit UI
# ===========================
st.title("📈 Stock Price Prediction App")

# Sidebar inputs
symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL")

# Historical data picker (up to today)
start = st.sidebar.date_input("Enter Start Date (historical):", dt.date(2020, 3, 4), max_value=TODAY)
end = st.sidebar.date_input("Enter End Date (historical):", dt.date.today(), min_value=start, max_value=TODAY)

# Future prediction date picker (can select any date after today, e.g., up to 2027)
future_date = st.sidebar.date_input(
    "Enter future date for prediction:",
    value=TODAY + dt.timedelta(days=1),
    min_value=TODAY + dt.timedelta(days=1),
    max_value=dt.date(2027, 12, 31)
)

# Sidebar buttons
if st.sidebar.button("Predict Future Price & Show Historical"):
    predict_stock(symbol, start, end, future_date)

if st.sidebar.button("View Close Price for Specific Date"):
    view_close_price(symbol)
