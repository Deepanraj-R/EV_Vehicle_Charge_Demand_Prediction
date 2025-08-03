import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
from PIL import Image
import time
import datetime
import pytz

# Page setup
st.set_page_config(page_title=" EV Forecast App", layout="wide")

# Pro Styling 
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;700&family=Orbitron:wght@500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #FFFFFF;
            animation: fadeIn 0.8s ease-in-out;
        }

        h1, h2, h3 {
            font-weight: 700;
            text-align: center;
            color: #00e6e6;
        }

        h1.main-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 44px;
            font-weight: 700;
            text-align: center;
            color: #00e6e6;
            margin-bottom: 1rem;
        }

        h3.custom-subheader {
            font-family: 'Orbitron', sans-serif;
            font-size: 28px;
            font-weight: 700;
            color: #00e6e6;
            margin-top: 2rem;
            text-align: left;
        }

        .metric-label, .metric-value {
            font-size: 18px;
        }

        .block-container {
            padding-top: 2rem;
        }

        .css-1v0mbdj img {
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0, 255, 255, 0.3);
        }

        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .stDownloadButton button {
            background-color: #00bcd4 !important;
            color: black !important;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.6em 1.8em;
            border: none;
            transition: all 0.3s ease;
        }
        .stDownloadButton button:hover {
            background-color: #0097a7 !important;
            color: white !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(145deg, #1a1a2e, #16213e);
            color: white;
            border-right: 3px solid #00e6e6;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #00e6e6 !important;         
        }
        section[data-testid="stSidebar"] label[data-baseweb="radio"] > div > div > span {
            color: #00e6e6 !important;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Loader Spinner 
with st.spinner("Loading model and data..."):
    time.sleep(0.5)

# Load Model and Data
@st.cache_data
def load_model():
    return joblib.load("forecasting_ev_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

model = load_model()
data = load_data()

# Header
st.markdown("<h1 class='main-title'>🔋 EV Adoption Trends in Washington</h1>", unsafe_allow_html=True)

st.image("EV_CAR 2.jpg", use_container_width=True, caption="Electric Vehicle Production Facility")

# Sidebar controls
st.sidebar.title("⚙️ Settings")
current_time = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))  
st.sidebar.markdown("---")
st.sidebar.markdown("🕒 <span style='font-size:16px; color:#00e6e6;'>Current Time (IST)</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span style='font-size:24px; font-weight:bold; color:#ffffff;'>{current_time.strftime('%I:%M:%S %p')}</span>", unsafe_allow_html=True)
st.sidebar.markdown("📅 <span style='font-size:16px; color:#00e6e6;'>Date</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span style='font-size:18px; color:#ffffff;'>{current_time.strftime('%A, %B %d, %Y')}</span>", unsafe_allow_html=True)
forecast_mode = st.sidebar.radio("Forecast Type", ["Monthly", "Yearly"])
forecast_value = st.sidebar.slider("Duration", 1, 5, 3)
forecast_months = forecast_value * 12 if forecast_mode == "Yearly" else forecast_value * 1
selected_county = st.sidebar.selectbox("Select County", sorted(data['County'].unique()))

# Date Range Filtering 
min_date, max_date = data['Date'].min(), data['Date'].max()
start_date, end_date = st.sidebar.date_input("Filter historical range", [min_date, max_date])

# Load selected county data
county_df = data[data['County'] == selected_county]
county_df = county_df[(county_df['Date'] >= pd.to_datetime(start_date)) & (county_df['Date'] <= pd.to_datetime(end_date))]

if county_df.empty:
    st.warning("No data available for the selected range.")
    st.stop()

# Prepare forecast
recent = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cum_ev = list(np.cumsum(recent))
months_since = county_df['months_since_start'].max()
last_date = county_df['Date'].max()
county_code = county_df['county_encoded'].iloc[0]

future_data = []
for i in range(1, forecast_months + 1):
    months_since += 1
    future_date = last_date + pd.DateOffset(months=i)
    lag1, lag2, lag3 = recent[-1], recent[-2], recent[-3]
    roll = np.mean([lag1, lag2, lag3])
    pct1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    slope = np.polyfit(range(len(cum_ev)), cum_ev, 1)[0]

    features = {
        'months_since_start': months_since,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll,
        'ev_total_pct_change_1': pct1,
        'ev_total_pct_change_3': pct3,
        'ev_growth_slope': slope
    }

    pred = model.predict(pd.DataFrame([features]))[0]
    future_data.append({'Date': future_date, 'Predicted EV Total': round(pred)})
    recent.append(pred)
    if len(recent) > 6: recent.pop(0)
    cum_ev.append(cum_ev[-1] + pred)
    if len(cum_ev) > 6: cum_ev.pop(0)

# Combine 
hist = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
hist['Cumulative'] = hist['Electric Vehicle (EV) Total'].cumsum()
hist['Source'] = 'Historical'
forecast_df = pd.DataFrame(future_data)
forecast_df['Cumulative'] = forecast_df['Predicted EV Total'].cumsum() + hist['Cumulative'].iloc[-1]
forecast_df['Source'] = 'Forecast'
combined = pd.concat([
    hist[['Date', 'Cumulative', 'Source']],
    forecast_df[['Date', 'Cumulative', 'Source']]
])

# Metrics
st.subheader(f"📍 Forecast Summary for {selected_county}")
total_hist = hist['Cumulative'].iloc[-1]
total_forecast = forecast_df['Cumulative'].iloc[-1]
growth_pct = ((total_forecast - total_hist) / total_hist) * 100 if total_hist else 0
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div style='padding:20px; background-color:#112d3a; border-radius:10px; text-align:center;'>
        <h4 style='color:#00e6e6;'>Current EV Count</h4>
        <p style='font-size:32px; font-weight:bold; color:#ffffff;'>{int(total_hist):,}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='padding:20px; background-color:#112d3a; border-radius:10px; text-align:center;'>
        <h4 style='color:#00e6e6;'>Projected EV Count</h4>
        <p style='font-size:32px; font-weight:bold; color:#00ffcc;'>{int(total_forecast):,}</p>
        <p style='font-size:16px; color:#ffa726;'>▲ {growth_pct:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

# Smart Summary
summary_html = f"""
<div style='
    background-color: #004d4d;
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    color: #00e6e6;
    font-weight: 500;
    text-align: center;
'>
    🔍 The forecast for <b>{selected_county}</b> shows an estimated <b>{int(total_forecast)} EVs</b> by <b>{forecast_df['Date'].max().date()}</b>,
    reflecting a <b>{growth_pct:.2f}%</b> increase from the current total.
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)

# Line Plot 
st.markdown("<h3 class='custom-subheader'>📈 Cumulative EV Trend</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 6))
for label, grp in combined.groupby("Source"):
    ax.plot(grp['Date'], grp['Cumulative'], label=label, marker='o')
ax.set_title(f"{selected_county} County EV Trend", color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.tick_params(colors='white')
ax.legend()
ax.set_facecolor("#0f2027")
fig.patch.set_facecolor("#0f2027")
st.pyplot(fig)

# Raw Forecast Plot
st.markdown("<h3 class='custom-subheader'>📉 Monthly Prediction and % Change</h3>", unsafe_allow_html=True)
forecast_df['Monthly % Change'] = forecast_df['Predicted EV Total'].pct_change() * 100
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(forecast_df['Date'], forecast_df['Predicted EV Total'], marker='o', label='Predicted EV')
ax3.set_ylabel("Predicted EVs", color='cyan')
ax3_2 = ax3.twinx()
ax3_2.plot(forecast_df['Date'], forecast_df['Monthly % Change'], color='orange', linestyle='--', marker='x', label='% Change')
ax3_2.set_ylabel("% Change", color='orange')
ax3.set_facecolor("#0f2027")
fig3.patch.set_facecolor("#0f2027")
ax3.tick_params(colors='white')
ax3_2.tick_params(colors='orange')
st.pyplot(fig3)

# CSV Download
st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False).encode('utf-8'), file_name=f"{selected_county}_forecast.csv", mime="text/csv")

# Save snapshot 
buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)
st.download_button("📸 Download Trend Snapshot", buf, file_name="forecast.png", mime="image/png")

# Footer
st.markdown("""
---
<div style='text-align: center; color: white;'>
    🚗 Forecast App | Designed by <b style='color: #00e6e6;'>Deepan Raj R</b> with | © 2025
</div>
""", unsafe_allow_html=True)
