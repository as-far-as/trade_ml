import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from etl.indicators import add_indicators

@st.cache_data(ttl=86400)
def get_index_tickers(index_name="SP500"):
    """Fetches stock lists from Wikipedia with high-precision table targeting."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    config = {
        "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "FTSE100": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "IBEX35": "https://en.wikipedia.org/wiki/IBEX_35"
    }
    
    try:
        url = config[index_name]
        response = requests.get(url, headers=headers, timeout=10)
        # Using a StringIO object is recommended but literal string works in most cases
        tables = pd.read_html(response.text, flavor='lxml')
        
        target_cols = ['TICKER', 'SYMBOL', 'EPIC', 'CODE']
        df = None
        ticker_col = None
        
        # Priority scan: find the first table that has one of our target columns
        for t in tables:
            # Convert all column names to uppercase string for matching
            cols_upper = [str(c).upper() for c in t.columns]
            for target in target_cols:
                if target in cols_upper:
                    df = t
                    # Get the actual column name (not uppercase)
                    ticker_col = t.columns[cols_upper.index(target)]
                    break
            if df is not None:
                break
        
        if df is None:
            # Emergency fallback based on current Wikipedia structures
            if index_name == "SP500": df = tables[0]; ticker_col = 'Symbol'
            elif index_name == "FTSE100": df = tables[6]; ticker_col = 'Ticker'
            elif index_name == "IBEX35": df = tables[2]; ticker_col = 'Ticker'
            else: return []

        tickers = df[ticker_col].astype(str).str.strip().tolist()

        # APPLY REGIONAL SUFFIXES
        if index_name == "FTSE100":
            return [f"{t}.L" if not t.endswith('.L') else t for t in tickers if len(t) < 10]
        if index_name == "IBEX35":
            # IBEX tickers often come as 'Ticker.MC' or just 'Ticker'
            return [f"{t}.MC" if not t.endswith('.MC') else t for t in tickers if len(t) < 10]
        
        # Clean S&P 500 (Yahoo uses - instead of . for classes like BRK.B)
        return [t.replace('.', '-') for t in tickers if len(t) < 10]

    except Exception as e:
        st.error(f"Scraper failed for {index_name}: {e}")
        return []

class DataFetcher:
    def __init__(self, ticker=None):
        self.ticker = ticker

    def fetch_historical_data(self, period="2y", interval="1d"):
        """Fetches historical OHLCV data from Yahoo Finance."""
        if not self.ticker:
            return pd.DataFrame()
        try:
            df = yf.download(self.ticker, period=period, interval=interval, progress=False)
            if df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:
            return pd.DataFrame()

    def add_indicators(self, df):
        return add_indicators(df)
