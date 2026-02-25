import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime 

class DataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker
        # Fixed: Removed manual session to comply with new yfinance requirements
        self.stock = yf.Ticker(ticker)

    def fetch_historical_data(self, period="1y", interval="1d"):
        """Fetches daily prices from Yahoo Finance."""
        try:
            df = self.stock.history(period=period, interval=interval)
            
            if df.empty:
                return pd.DataFrame()

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            return df
        except Exception as e:
            st.error(f"Error fetching {self.ticker}: {e}")
            return pd.DataFrame()

    def add_indicators(self, df):
        """Adds RSI, Moving Averages, and Volatility."""
        if df.empty: return df
            
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
    
    def generate_mock_data(self):
        """Generates realistic synthetic stock data when API is offline."""
        date_rng = pd.date_range(end=datetime.now(), periods=100, freq='D')
        start_price = np.random.uniform(50, 200)
        changes = np.random.normal(0.001, 0.02, 100) 
        price_path = start_price * (1 + changes).cumprod()
        
        return pd.DataFrame({
            'Close': price_path,
            'Open': price_path * np.random.uniform(0.98, 1.02, 100),
            'High': price_path * 1.03,
            'Low': price_path * 0.97,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=date_rng)