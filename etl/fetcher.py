import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime 

class DataFetcher:
    def __init__(self, ticker=None):
        self.ticker = ticker
        if ticker:
            self.stock = yf.Ticker(ticker)

    @st.cache_data(ttl=86400) # Cache the ticker lists for 24 hours
    @staticmethod
    def get_index_tickers(index_name="SP500"):
        """
        Fetches live index constituents from Wikipedia.
        Static method avoids Streamlit's UnhashableParamError by ignoring 'self'.
        """
        try:
            if index_name == "SP500":
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                table = pd.read_html(url)[0]
                # S&P 500 uses 'Symbol'
                return table['Symbol'].tolist()
            
            elif index_name == "FTSE100":
                url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
                table = pd.read_html(url)[4] # Typically table 4 for constituents
                # FTSE uses 'EPIC' and needs '.L' suffix for Yahoo
                tickers = table['EPIC'].tolist()
                return [f"{t}.L" for t in tickers]
            
            elif index_name == "IBEX35":
                url = "https://en.wikipedia.org/wiki/IBEX_35"
                table = pd.read_html(url)[1] # Typically table 1
                # IBEX uses 'Ticker' and needs '.MC' suffix for Yahoo
                tickers = table['Ticker'].tolist()
                return [f"{t}.MC" for t in tickers]
                
            return []
        except Exception as e:
            st.error(f"Scraping Error for {index_name}: {e}")
            return []

    def fetch_historical_data(self, period="1y", interval="1d"):
        """Fetches daily prices from Yahoo Finance for a single ticker."""
        if not self.ticker:
            return pd.DataFrame()
        try:
            df = self.stock.history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            print(f"Error fetching {self.ticker}: {e}")
            return pd.DataFrame()

    def add_indicators(self, df):
        """Adds RSI, Moving Averages, and Volatility to any dataframe."""
        if df.empty or len(df) < 20:
            return df
            
        # 1. RSI (14-period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. Moving Averages
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # 3. Volatility (21-day Standard Deviation of Returns)
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=21).std()
        
        return df.dropna()
    
    def generate_mock_data(self):
        """Generates realistic synthetic stock data when API/Network fails."""
        date_rng = pd.date_range(end=datetime.now(), periods=100, freq='D')
        start_price = np.random.uniform(50, 200)
        changes = np.random.normal(0.0005, 0.02, 100) 
        price_path = start_price * (1 + changes).cumprod()
        
        df = pd.DataFrame({
            'Close': price_path,
            'Open': price_path * np.random.uniform(0.99, 1.01, 100),
            'High': price_path * 1.02,
            'Low': price_path * 0.98,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=date_rng)
        
        return df