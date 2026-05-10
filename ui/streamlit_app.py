import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Vietnam Stock Forecast", layout="wide")
st.title("Vietnam Stock Price Forecast Tool")
st.caption("Powered by CNN Deep Learning Model — Task 2.3")

# Sidebar config 
with st.sidebar:
    st.header("Configuration")
    ticker     = st.text_input("Stock Ticker", value="HPG")
    input_mode = st.radio("Input Mode",
                          ["Close prices only", "Full OHLCV"])
    st.markdown("---")
    st.markdown("**API Status**")
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if health.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Error")
    except:
        st.error("API Offline — start FastAPI first")

# Input table
st.subheader("Step 1 — Enter Recent 30 Trading Days")
st.info("Feature order must be: Open, High, Low, Close, Volume")

if input_mode == "Close prices only":
    df = pd.DataFrame({
        "Day":   [f"Day {i+1}" for i in range(30)],
        # Example input 
        "Close": [
            21000, 20900, 20700, 20800, 21100,
            21200, 21300, 21400, 21500, 21600,
            21500, 21700, 21800, 21900, 22000,
            21800, 21900, 22100, 22000, 21900,
            21800, 21700, 21800, 21900, 22000,
            21900, 21800, 21700, 21600, 21500
        ]
    })
    edited_df = st.data_editor(df, use_container_width=True,
                                num_rows="fixed")
    try:
        closes = edited_df["Close"].astype(float).tolist()
        valid  = len(closes) == 30
        # Replicate Close as OHLC, keep volume fixed
        # Feature order: Open, High, Low, Close, Volume
        ohlcv  = [[p, p*1.01, p*0.99, p, 500000] for p in closes]
    except:
        valid = False

else:
    df = pd.DataFrame({
        "Day":    [f"Day {i+1}" for i in range(30)],
        # Example input 
        "Open":   [20800,21000,20900,20700,20800,21100,21200,21300,21400,21500,21600,21500,21700,21800,21900,22000,21800,21900,22100,22000,21900,21800,21700,21800,21900,22000,21900,21800,21700,21600],
        "High":   [21200,21300,21100,21000,21200,21400,21500,21600,21700,21800,21900,21800,22000,22100,22200,22300,22100,22200,22400,22300,22200,22100,22000,22100,22200,22300,22200,22100,22000,21900],
        "Low":    [20600,20700,20500,20400,20600,20900,21000,21100,21200,21300,21400,21300,21500,21600,21700,21800,21600,21700,21900,21800,21700,21600,21500,21600,21700,21800,21700,21600,21500,21400],
        "Close":  [21000,20900,20700,20800,21100,21200,21300,21400,21500,21600,21500,21700,21800,21900,22000,21800,21900,22100,22000,21900,21800,21700,21800,21900,22000,21900,21800,21700,21600,21500],
        "Volume": [850000,920000,780000,810000,870000,900000,860000,830000,890000,910000,875000,840000,920000,930000,950000,880000,860000,870000,890000,910000,895000,850000,840000,860000,875000,890000,870000,855000,840000,825000]
    })
    edited_df = st.data_editor(df, use_container_width=True,
                                num_rows="fixed")
    try:
        vals  = edited_df[["Open","High","Low","Close","Volume"]].astype(float)
        valid = vals.shape == (30, 5)
        ohlcv = vals.values.tolist()
    except:
        valid = False

# Predict 
st.subheader("Step 2 — Run Forecast")

if st.button("Predict Next 7 Days", type="primary"):
    if not valid:
        st.error("Please fill all 30 rows with valid numbers.")
    else:
        with st.spinner("Calling API..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"ticker": ticker, "instances": ohlcv},
                    timeout=10
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. "
                         "Run: uvicorn app:app --reload")
                st.stop()

        if response.status_code != 200:
            st.error(f"API error {response.status_code}: "
                     f"{response.json().get('detail', '')}")
        else:
            preds         = response.json()["prediction"]
            close_history = [row[3] for row in ohlcv]
            current_price = close_history[-1]

            # Results 
            st.subheader(f"Step 3 — Forecast Results for {ticker}")

            # Metrics row
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price",
                        f"{current_price:,.0f} VND")
            col2.metric(f"Predicted Day +7",
                        f"{preds[-1]:,.0f} VND",
                        f"{preds[-1] - current_price:+,.0f} VND")
            col3.metric("Average Forecast",
                        f"{np.mean(preds):,.0f} VND",
                        f"{np.mean(preds) - current_price:+,.0f} VND")

            # Combined chart: history + forecast
            hist_days = [f"Day {i+1}" for i in range(30)]
            fore_days = [f"T+{i+1}" for i in range(len(preds))]

            fig = go.Figure()

            # Historical close prices
            fig.add_trace(go.Scatter(
                x=hist_days, y=close_history,
                name="Historical Close",
                line=dict(color="#1f77b4", width=2)
            ))

            # Connector from last historical to first forecast
            fig.add_trace(go.Scatter(
                x=[hist_days[-1], fore_days[0]],
                y=[current_price, preds[0]],
                name="",
                line=dict(color="#2ecc71", width=2, dash="dot"),
                showlegend=False
            ))

            # Forecast prices
            fig.add_trace(go.Scatter(
                x=fore_days, y=preds,
                name="Forecast",
                line=dict(color="#2ecc71", width=2, dash="dash"),
                mode="lines+markers",
                marker=dict(size=8)
            ))

            fig.update_layout(
                title=f"{ticker} — Historical & 7-Day Forecast",
                xaxis_title="Trading Day",
                yaxis_title="Price (VND)",
                hovermode="x unified",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            result_df = pd.DataFrame({
                "Day":             fore_days,
                "Predicted Close (VND)": [f"{p:,.0f}" for p in preds],
                "Change from Current":   [f"{p-current_price:+,.0f}" for p in preds],
                "Change (%)":            [f"{(p-current_price)/current_price*100:+.2f}%" for p in preds]
            })
            st.dataframe(result_df, use_container_width=True,
                          hide_index=True)