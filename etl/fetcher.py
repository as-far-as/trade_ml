import pandas as pd
import numpy as np
import requests
import streamlit as st

class DataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker
        self.api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_historical_data(self, period="1y", interval="daily"):
        """
        Fetches daily price data. 
        Note: Free tier allows 25 requests/day.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.ticker,
            "outputsize": "compact", # 'compact' returns last 100 days; 'full' returns 20 years
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                print(f"Error fetching {self.ticker}: {data.get('Note', 'Check API Key')}")
                return pd.DataFrame()

            # Parse JSON to DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            # Rename columns to match your original yfinance style
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            return df
            
        except Exception as e:
            print(f"Connection Error: {e}")
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

if __name__ == "__main__":
    # For local testing, ensure you have an environment variable or hardcoded key
    # fetcher = DataFetcher("AAPL")
    pass