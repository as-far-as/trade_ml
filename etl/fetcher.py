import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import datetime 

class DataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker
        self.api_key = st.secrets.get("FMP_API_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def fetch_historical_data(self):
        """Fetches daily prices from FMP (250 req/day limit)."""
        url = f"{self.base_url}/historical-price-full/{self.ticker}?apikey={self.api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            
            if "historical" not in data:
                return pd.DataFrame()

            df = pd.DataFrame(data["historical"])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Standardize column names for your indicators
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df
        except:
            return pd.DataFrame()

    def fetch_fundamentals(self):
        """Extracts key ratios using the OVERVIEW endpoint."""
        params = {
            "function": "OVERVIEW",
            "symbol": self.ticker,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            info = response.json()
            
            # Map Alpha Vantage keys to your existing expected keys
            return {
                "trailingPE": info.get("TrailingPE"),
                "forwardPE": info.get("ForwardPE"),
                "priceToBook": info.get("PriceToBookRatio"),
                "debtToEquity": info.get("DebtToEquityRatio"),
                "revenueGrowth": info.get("QuarterlyRevenueGrowthYOY")
            }
        except:
            return {}

    def add_indicators(self, df):
        """Adds RSI, Moving Averages, and Volatility."""
        if df.empty:
            return df
            
        # 1. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. Moving Averages
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # 3. Volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=21).std()
        
        return df.dropna()
    
    def generate_mock_data(self):
        """Generates realistic synthetic stock data when API is offline."""
        date_rng = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Start at a random price between 50 and 200
        start_price = np.random.uniform(50, 200)
        # Create a random walk
        changes = np.random.normal(0.001, 0.02, 100) 
        price_path = start_price * (1 + changes).cumprod()
        
        df = pd.DataFrame({
            'Close': price_path,
            'Open': price_path * np.random.uniform(0.98, 1.02, 100),
            'High': price_path * 1.03,
            'Low': price_path * 0.97,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=date_rng)
        
        return df

if __name__ == "__main__":
    # For local testing, ensure you have an environment variable or hardcoded key
    # fetcher = DataFetcher("AAPL")
    pass