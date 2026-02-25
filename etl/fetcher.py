import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
from datetime import datetime 

# --- STANDALONE CACHED FUNCTIONS ---

@st.cache_data(ttl=86400)

def get_index_tickers(index_name="SP500"):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if index_name == "SP500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            table_idx = 0
        elif index_name == "FTSE100":
            url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
            table_idx = 4
        elif index_name == "IBEX35":
            url = "https://en.wikipedia.org/wiki/IBEX_35"
            table_idx = 1
        
        response = requests.get(url, headers=headers)
        df = pd.read_html(response.text, flavor='lxml')[table_idx]
        
        # FUZZY COLUMN SEARCH: Find a column that looks like Ticker/Symbol/EPIC
        target_cols = ['Symbol', 'Ticker', 'EPIC', 'Ticker symbol', 'Code']
        found_col = None
        for c in target_cols:
            if c in df.columns:
                found_col = c
                break
        
        if not found_col:
            return []

        tickers = df[found_col].tolist()
        
        # Clean and suffix
        if index_name == "FTSE100": return [f"{str(t).strip()}.L" for t in tickers]
        if index_name == "IBEX35": return [f"{str(t).strip()}.MC" for t in tickers]
        return [str(t).strip().replace('.', '-') for t in tickers]

    except Exception as e:
        st.error(f"Scraping Error: {e}")
        return []

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