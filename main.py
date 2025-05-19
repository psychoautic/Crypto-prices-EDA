from ollama import Llama3
import prophet
import customtkinter as ctk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cctx as cx

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


# Example usage (remove or modify as needed)
llama = Llama3()
print(llama)