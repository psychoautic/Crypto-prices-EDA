# from ollama import Llama3
from prophet import Prophet
import customtkinter as ctk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import cctx as cx

AVAX = pd.read_csv('data/Avalanche_hist.csv')
BUSD = pd.read_csv('data/BinanceUSD_hist.csv')
BTC = pd.read_csv('data/Bitcoin_hist.csv')
BNB = pd.read_csv('data/BNB_hist.csv')
ADA = pd.read_csv('data/Cardano_hist.csv')
ATOM = pd.read_csv('data/Cosmos_hist.csv')
CRO = pd.read_csv('data/Cronos_hist.csv')
DAI = pd.read_csv('data/Dai_hist.csv')
DOGE = pd.read_csv('data/Dogecoin_hist.csv')
ETH = pd.read_csv('data/Ethereum_hist.csv')
DOT = pd.read_csv('data/Polkadot_hist.csv')
POL = pd.read_csv('data/Polygon_hist.csv')
SHIB = pd.read_csv('data/Shibainu_hist.csv')
SOL = pd.read_csv('data/Solana_hist.csv')
LUNA = pd.read_csv('data/Terra_hist.csv')
USDT = pd.read_csv('data/Tether_hist.csv')
USDC = pd.read_csv('data/USDCoin_hist.csv')
WBTC = pd.read_csv('data/WrappedBitcoin_hist.csv')
XRP = pd.read_csv('data/XRP_hist.csv')

BTC_prophet = BTC.rename(columns={'date': 'ds', 'close_price': 'y'})

# Step 2: Log-transform the 'y' values to avoid negative predictions
BTC_prophet = BTC_prophet[BTC_prophet['y'] > 0]  # Just in case
BTC_prophet['y'] = np.log(BTC_prophet['y'])

# Step 3: Fit the Prophet model
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
m.fit(BTC_prophet)

# Step 4: Make future dataframe and predict
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Step 5: Reverse the log scale to get actual price predictions
forecast['yhat'] = np.exp(forecast['yhat'])
forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])

# Step 6: Print results
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].sample(10))

actual = BTC.rename(columns={'date': 'ds', 'close_price': 'y'})
actual = actual[actual['y'] > 0]  # same filtering as before

# Ensure 'ds' columns are datetime
actual['ds'] = pd.to_datetime(actual['ds'])
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Plotting
plt.figure(figsize=(12, 6))

# Actual prices
plt.plot(actual['ds'], actual['y'], label='Actual BTC Price', color='black')

# Forecasted prices
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted BTC Price', color='blue')

# Confidence intervals as shaded region
plt.fill_between(
    forecast['ds'],
    forecast['yhat_lower'],
    forecast['yhat_upper'],
    color='skyblue',
    alpha=0.4,
    label='Confidence Interval'
)

plt.title("BTC Price: Actual vs Prophet Forecast")
plt.xlabel("Date")
plt.ylabel("BTC Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()