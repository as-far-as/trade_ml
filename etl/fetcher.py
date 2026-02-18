import yfinance as yf
import pandas as pd
import numpy as np

class DataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def fetch_historical_data(self, period="2y", interval="1d"):
        """Extracts historical price and volume data."""
        df = self.stock.history(period=period, interval=interval)
        return df

    def fetch_fundamentals(self):
        """Extracts key ratios: P/E, Forward P/E, and Earnings info."""
        info = self.stock.info
        fundamentals = {
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "debtToEquity": info.get("debtToEquity"),
            "revenueGrowth": info.get("revenueGrowth")
        }
        return fundamentals

    def add_indicators(self, df):
        """Transformation: Adds RSI and Moving Averages."""
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Add 50-day and 200-day Moving Averages
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate Volatility (Standard Deviation of returns)
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=21).std()
        
        return df.dropna()

if __name__ == "__main__":
    # Test it out
    fetcher = DataFetcher("AAPL")
    prices = fetcher.fetch_historical_data()
    data_with_indicators = fetcher.add_indicators(prices)
    
    print(f"--- Data for AAPL ---")
    print(data_with_indicators.tail(5)) # Show last 5 days
    print(f"\nFundamental Ratios: {fetcher.fetch_fundamentals()}")