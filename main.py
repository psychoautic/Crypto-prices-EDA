import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import customtkinter as ctk
import ollama
import json
import re
import ast

tickerToFullname = {
    "AVAX": "Avalanche",
    "BUSD": "BinanceUSD",
    "BTC": "Bitcoin",
    "BNB": "BNB",
    "ADA": "Cardano",
    "ATOM": "Cosmos",
    "CRO": "Cronos",
    "DAI": "Dai",
    "DOGE": "Dogecoin",
    "ETH": "Ethereum",
    "DOT": "Polkadot",
    "POL": "Polygon",
    "SHIB": "Shibainu",
    "SOL": "Solana",
    "LUNA": "Terra",
    "USDT": "Tether",
    "USDC": "USDCoin",
    "WBTC": "WrappedBitcoin",
    "XRP": "XRP"
}

data = {}
for ticker, fullname in tickerToFullname.items():
    path = f"data/{fullname}_hist.csv"
    try:
        data[ticker] = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")

def llamaQuery(prompt: str) -> str:
    try:
        response = ollama.chat("llama3", messages=[{"role": "user", "content": prompt}])
        return response.message["content"]
    except Exception as e:
        return f"LLaMA query failed: {e}"

def prophetPredict(ticker, feature, periods=365):
    if ticker not in data:
        return f"Ticker {ticker} not found."
    df = data[ticker].copy()
    if feature not in df.columns:
        available = ', '.join(df.columns)
        return (
            f"Feature '{feature}' not found in {ticker} data.\n"
            f"Available features: {available}"
        )
    dfProphet = df.rename(columns={'date': 'ds', feature: 'y'})
    dfProphet = dfProphet[dfProphet['y'] > 0].copy()
    dfProphet['y'] = np.log(dfProphet['y'])
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(dfProphet)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def compareCryptos(ticker1, ticker2, feature, periods=365):
    f1 = prophetPredict(ticker1, feature, periods)
    f2 = prophetPredict(ticker2, feature, periods)
    if isinstance(f1, str):
        return f1
    if isinstance(f2, str):
        return f2
    plt.figure(figsize=(12, 6))
    plt.plot(f1['ds'], f1['yhat'], label=f"{ticker1} forecast {feature}")
    plt.plot(f2['ds'], f2['yhat'], label=f"{ticker2} forecast {feature}")
    plt.title(f"Forecast Comparison of {feature} between {ticker1} and {ticker2}")
    plt.xlabel("Date")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return "Comparison plot displayed."

def extractJsonFromResponse(response):
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if match:
        return match.group(0)
    return None

def explainForecast(ticker, feature, forecast):
    lastRow = forecast.iloc[-1]
    firstRow = forecast.iloc[0]
    change = lastRow['yhat'] - firstRow['yhat']
    pctChange = (change / firstRow['yhat']) * 100 if firstRow['yhat'] != 0 else 0
    return (
        f"Prediction for {ticker} feature '{feature}':\n"
        f"- Start: {firstRow['ds'].date()} value ≈ {firstRow['yhat']:.2f}4\n"
        f"- End: {lastRow['ds'].date()} value ≈ {lastRow['yhat']:.2f}\n"
        f"- Change: {change:.2f} ({pctChange:.2f}%) over the period.\n"
    )

def safeJsonLoads(s):
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            sFixed = s
            if s.count('{') > s.count('}'):
                sFixed += '}' * (s.count('{') - s.count('}'))
            if s.count('[') > s.count(']'):
                sFixed += ']' * (s.count('[') - s.count(']'))
            try:
                return json.loads(sFixed)
            except Exception:
                return None

def handleUserQuery(userInput):
    prompt = (
        f"You are a helpful crypto assistant. User input: '{userInput}'.\n"
        "Extract the crypto ticker(s), feature(s) (like close_price, volume, market_cap, etc.), "
        "and what prediction or comparison they want for the next 365 days. "
        "If the user asks for a specific date or range, include it as a 'date' or 'date_range' field in the JSON.\n"
        "Format your answer as JSON like: "
        '{"action":"predict" or "compare","tickers":["BTC"],"feature":"close_price","date":"2023-12-21"}\n'
        "If unclear, ask for clarification."
    )
    response = llamaQuery(prompt)
    jsonStr = extractJsonFromResponse(response)
    if not jsonStr:
        return response.strip()
    intent = safeJsonLoads(jsonStr)
    if not intent:
        return f"Failed to parse JSON in LLaMA response: {jsonStr}"
    action = intent.get("action")
    feature = intent.get("feature")
    if not feature:
        features = intent.get("features")
        if isinstance(features, list) and features:
            feature = features[0]
        else:
            feature = "close_price"
    tickers = [t.upper() for t in intent.get("tickers", [])]
    dateInfo = intent.get("date", None)
    dateRange = intent.get("date_range", None)
    dateStart = dateEnd = None
    if isinstance(dateInfo, dict):
        dateStart = dateInfo.get("start")
        dateEnd = dateInfo.get("end")
    elif isinstance(dateInfo, str):
        dateStart = dateEnd = dateInfo
    elif isinstance(dateRange, list):
        if len(dateRange) == 2 and all(isinstance(x, str) for x in dateRange):
            dateStart, dateEnd = dateRange
        elif len(dateRange) > 0 and isinstance(dateRange[0], dict):
            dr = dateRange[0]
            dateStart = dr.get("start")
            dateEnd = dr.get("end")
    elif isinstance(dateRange, dict):
        dateStart = dateRange.get("start")
        dateEnd = dateRange.get("end")
    if action == "compare" and len(tickers) == 2 and dateStart and dateEnd:
        f1 = prophetPredict(tickers[0], feature)
        f2 = prophetPredict(tickers[1], feature)
        if isinstance(f1, str):
            return f1
        if isinstance(f2, str):
            return f2
        f1['ds'] = pd.to_datetime(f1['ds'])
        f2['ds'] = pd.to_datetime(f2['ds'])
        startDt = pd.to_datetime(dateStart)
        endDt = pd.to_datetime(dateEnd)
        f1Start = f1.iloc[(f1['ds'] - startDt).abs().argsort()[:1]]
        f1End = f1.iloc[(f1['ds'] - endDt).abs().argsort()[:1]]
        f2Start = f2.iloc[(f2['ds'] - startDt).abs().argsort()[:1]]
        f2End = f2.iloc[(f2['ds'] - endDt).abs().argsort()[:1]]
        btcChange = f1End['yhat'].values[0] - f1Start['yhat'].values[0]
        solChange = f2End['yhat'].values[0] - f2Start['yhat'].values[0]
        btcPct = (btcChange / f1Start['yhat'].values[0]) * 100
        solPct = (solChange / f2Start['yhat'].values[0]) * 100
        explanation = (
            f"Between {startDt.date()} and {endDt.date()}:\n"
            f"- {tickers[0]} ({feature}): {f1Start['yhat'].values[0]:.2f} → {f1End['yhat'].values[0]:.2f} "
            f"({btcChange:+.2f}, {btcPct:+.2f}%)\n"
            f"- {tickers[1]} ({feature}): {f2Start['yhat'].values[0]:.2f} → {f2End['yhat'].values[0]:.2f} "
            f"({solChange:+.2f}, {solPct:+.2f}%)\n"
        )
        if btcPct > solPct:
            explanation += f"{tickers[0]} outperformed {tickers[1]} in this interval."
        elif solPct > btcPct:
            explanation += f"{tickers[1]} outperformed {tickers[0]} in this interval."
        else:
            explanation += "Both performed similarly in this interval."
        return explanation
    if action == "predict" and len(tickers) == 1:
        forecast = prophetPredict(tickers[0], feature)
        if isinstance(forecast, str):
            return forecast
        if dateStart and dateEnd and dateStart != dateEnd:
            try:
                forecast['ds'] = pd.to_datetime(forecast['ds'])
                startDt = pd.to_datetime(dateStart)
                endDt = pd.to_datetime(dateEnd)
                rowStart = forecast.iloc[(forecast['ds'] - startDt).abs().argsort()[:1]]
                rowEnd = forecast.iloc[(forecast['ds'] - endDt).abs().argsort()[:1]]
                valueStart = rowStart['yhat'].values[0]
                valueEnd = rowEnd['yhat'].values[0]
                change = valueEnd - valueStart
                pctChange = (change / valueStart) * 100 if valueStart != 0 else 0
                return (
                    f"Prediction for {tickers[0]} {feature} from {startDt.date()} to {endDt.date()}:\n"
                    f"- Start: {valueStart:.2f}\n"
                    f"- End: {valueEnd:.2f}\n"
                    f"- Change: {change:+.2f} ({pctChange:+.2f}%)"
                )
            except Exception as e:
                return f"Could not find prediction for range {dateStart} to {dateEnd}: {e}"
        if dateStart:
            try:
                targetDate = pd.to_datetime(dateStart)
                forecast['ds'] = pd.to_datetime(forecast['ds'])
                closestRow = forecast.iloc[(forecast['ds'] - targetDate).abs().argsort()[:1]]
                value = closestRow['yhat'].values[0]
                lower = closestRow['yhat_lower'].values[0]
                upper = closestRow['yhat_upper'].values[0]
                return (f"Prediction for {tickers[0]} {feature} on {targetDate.date()}:\n"
                        f"Value ≈ {value:.2f} (range: {lower:.2f} - {upper:.2f})")
            except Exception as e:
                return f"Could not find prediction for date {dateStart}: {e}"
        plt.figure(figsize=(10, 5))
        plt.plot(forecast['ds'], forecast['yhat'], label=f"{tickers[0]} {feature} forecast")
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
        plt.title(f"{tickers[0]} {feature} Forecast")
        plt.xlabel("Date")
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        explanation = explainForecast(tickers[0], feature, forecast)
        return explanation
    return (
        "Sorry, I couldn't understand the request. "
        "Please specify one or two tickers and a feature present in the data."
    )

class CryptoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Crypto Price Chatbot + LLaMA NLP")
        self.geometry("700x600")
        self.label = ctk.CTkLabel(self, text="Crypto Chatbot (powered by LLaMA):")
        self.label.pack(pady=8)
        self.chatbox = ctk.CTkTextbox(self, width=650, height=350, state="disabled", wrap="word")
        self.chatbox.pack(pady=8)
        self.inputBox = ctk.CTkTextbox(self, width=650, height=50)
        self.inputBox.pack(pady=8)
        btnFrame = ctk.CTkFrame(self)
        btnFrame.pack(pady=5)
        self.sendBtn = ctk.CTkButton(btnFrame, text="Send", command=self.onSend)
        self.sendBtn.pack(side="left", padx=10)
        self.clearBtn = ctk.CTkButton(btnFrame, text="Clear Conversation", command=self.clearChat)
        self.clearBtn.pack(side="left", padx=10)

    def appendChat(self, sender, message):
        self.chatbox.configure(state="normal")
        self.chatbox.insert("end", f"{sender}: {message}\n\n")
        self.chatbox.see("end")
        self.chatbox.configure(state="disabled")

    def onSend(self):
        userMsg = self.inputBox.get("0.0", "end").strip()
        if not userMsg:
            return
        self.appendChat("You", userMsg)
        self.inputBox.delete("0.0", "end")
        self.appendChat("Bot", "Processing...")
        self.update_idletasks()
        try:
            answer = handleUserQuery(userMsg)
        except Exception as e:
            answer = f"Error: {e}"
        self.chatbox.configure(state="normal")
        lines = self.chatbox.get("0.0", "end").splitlines()
        if len(lines) >= 2 and lines[-2].startswith("Bot: Processing..."):
            self.chatbox.delete(f"end-3l", "end-1l")
        self.chatbox.configure(state="disabled")
        self.appendChat("Bot", answer)

    def clearChat(self):
        self.chatbox.configure(state="normal")
        self.chatbox.delete("0.0", "end")
        self.chatbox.configure(state="disabled")
        self.inputBox.delete("0.0", "end")

if __name__ == "__main__":
    app = CryptoApp()
    app.mainloop()
