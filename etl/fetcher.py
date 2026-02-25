import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
from datetime import datetime 

# --- STANDALONE CACHED FUNCTIONS ---

@st.cache_data(ttl=86400)
def get_index_tickers(index_name="SP500"):
    """
    Fetches live index constituents with browser-impersonation headers
    to prevent connection blocks.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        if index_name == "SP500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            # We use requests to get the HTML with headers, then pass to pandas
            response = requests.get(url, headers=headers)
            table = pd.read_html(response.text, flavor='lxml')[0]
            return table['Symbol'].tolist()
        
        elif index_name == "FTSE100":
            url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
            response = requests.get(url, headers=headers)
            table = pd.read_html(response.text, flavor='lxml')[4]
            tickers = table['EPIC'].tolist()
            return [f"{t}.L" for t in tickers]
        
        elif index_name == "IBEX35":
            url = "https://en.wikipedia.org/wiki/IBEX_35"
            response = requests.get(url, headers=headers)
            table = pd.read_html(response.text, flavor='lxml')[1]
            tickers = table['Ticker'].tolist()
            return [f"{t}.MC" for t in tickers]
            
        return []
    except Exception as e:
        # If scraping fails, we provide a small fallback list so the app doesn't crash
        st.error(f"Scraping Error: {e}")
        fallbacks = {"SP500": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], "FTSE100": ["AZN.L", "HSBA.L"], "IBEX35": ["SAN.MC", "ITX.MC"]}
        return fallbacks.get(index_name, [])

# --- DATA PROCESSOR CLASS ---
class DataFetcher:
    def __init__(self, ticker=None):
        self.ticker = ticker
        if ticker:
            self.stock = yf.Ticker(ticker)

    def add_indicators(self, df):
        if df.empty or len(df) < 20: return pd.DataFrame()
        
        # Calculate Technicals
        df = df.copy() # Avoid fragmentation warnings
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=21).std()
        
        return df.dropna()