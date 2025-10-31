# ===========================
# Stock Price Prediction App (Enhanced UI, Main Page Outputs)
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
import plotly.graph_objects as go

# ---------------------------
# Constants
# ---------------------------
TODAY = dt.date.today()  # current date

# ===========================
# Function: Fetch historical data
# ===========================
@st.cache_data
def get_stock_data(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance."""
    if start >= end:
        st.error("❌ Start date must be before end date.")
        return None
    if end > TODAY:
        end = TODAY

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
    """Allow user to select a date in sidebar and display close price info on the main page."""
    st.subheader("📅 View Close Price for a Specific Date")

    # Sidebar inputs
    with st.sidebar.expander("📅 Close Price Lookup", expanded=True):
        selected_date = st.date_input(
            "Select date:",
            value=TODAY - dt.timedelta(days=1),
            max_value=TODAY
        )
        show_price = st.button("🔍 Show Close Price")

    # Display results on main page
    if show_price:
        try:
            df = yf.download(symbol, start=selected_date, end=selected_date + dt.timedelta(days=1), progress=False)
            if df.empty:
                st.warning(f"⚠️ No trading data found for {symbol} on {selected_date}.")
                return

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            open_price = float(df["Open"].iloc[0])
            high_price = float(df["High"].iloc[0])
            low_price = float(df["Low"].iloc[0])
            close_price = float(df["Close"].iloc[0])

            st.success(f"**Close Price of {symbol} on {selected_date}: ${close_price:.2f}**")
            st.markdown(f"**Open:** ${open_price:.2f} | **High:** ${high_price:.2f} | **Low:** ${low_price:.2f}")

            # Compact single-day plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=['Open','High','Low','Close'],
                y=[open_price, high_price, low_price, close_price],
                mode='lines+markers',
                line=dict(color='orange'),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title=f"{symbol} Prices on {selected_date}",
                yaxis_title="Price ($)",
                xaxis_title="Parameter",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error fetching close price: {e}")

# ===========================
# Function: Train LSTM and predict future price
# ===========================
def predict_stock(symbol: str, start: dt.date, end: dt.date, future_date: dt.date):
    """Shows historical data and predicts future stock price using LSTM."""
    df = get_stock_data(symbol, start, end)
    if df is None:
        return

    # ---------------------------
    # Tabs for organization
    # ---------------------------
    tab1, tab2 = st.tabs(["📊 Historical Data", "🔮 Prediction"])

    # ---------------------------
    # Historical Data Tab
    # ---------------------------
    with tab1:
        st.subheader("📊 Historical Data")
        st.dataframe(df)

        st.subheader("📈 Closing Price Graph")

        # Simple line graph of closing prices
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',  # just a line
            name='Close Price',
            line=dict(color='blue')
        ))

        fig_hist.update_layout(
            title=f"{symbol} Closing Prices ({start} to {end})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            hovermode="x unified",
            legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0)')
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    # ---------------------------
    # LSTM Prediction
    # ---------------------------
    if "Close" not in df.columns:
        st.error("⚠️ 'Close' column not found. Check stock symbol.")
        return

    data = df[['Close']]
    dataset = data.values
    lookback = min(60, len(dataset))
    scaled_data = MinMaxScaler(feature_range=(0,1)).fit_transform(dataset)

    # Create sequences
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i,0])
        y_train.append(scaled_data[i,0])
    if len(x_train) == 0:
        x_train.append(scaled_data[-lookback:])
        y_train.append(scaled_data[-1,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

    # Build and train LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("⏳ Training model..."):
        model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

    # ---------------------------
    # Prediction Tab
    # ---------------------------
    with tab2:
        st.subheader(f"🔮 Predicted Close Price for {future_date}")
        last_days = dataset[-lookback:]
        scaler_last = MinMaxScaler(feature_range=(0,1))
        last_scaled = scaler_last.fit_transform(last_days)
        X_future = np.array([last_scaled])
        X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1],1))
        pred_scaled = model.predict(X_future)
        pred_price = scaler_last.inverse_transform(pred_scaled)
        st.success(f"${pred_price[0][0]:.2f}")

# ===========================
# Streamlit App
# ===========================
st.title("📈 Stock Price Prediction App (Enhanced UI)")

# Sidebar: inputs only
with st.sidebar.expander("📊 Historical & Prediction Inputs", expanded=True):
    symbol = st.text_input("Stock Symbol:", value="AAPL")
    start = st.date_input("Start Date:", dt.date(2020,3,4), max_value=TODAY)
    end = st.date_input("End Date:", dt.date.today(), min_value=start, max_value=TODAY)
    future_date = st.date_input(
        "Future Prediction Date:",
        value=TODAY + dt.timedelta(days=1),
        min_value=TODAY + dt.timedelta(days=1),
        max_value=dt.date(2027,12,31)
    )
    run_analysis = st.button("Run Analysis")

# Main page: outputs
if run_analysis:
    predict_stock(symbol, start, end, future_date)

# Always show Close Price Lookup (main page)
view_close_price(symbol)
