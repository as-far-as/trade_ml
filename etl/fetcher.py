import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

class DataFetcher:
    def __init__(self):
        pass

    @st.cache_data(ttl=86400)
    @staticmethod  # <--- Add this decorator!
    def get_index_tickers(index_name="SP500"):
        try:
            # Note: We removed 'self' from the arguments below
            if index_name == "SP500":
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                table = pd.read_html(url)[0]
                return table['Symbol'].tolist()
            elif index_name == "FTSE100":
                url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
                table = pd.read_html(url)[4] # Usually table 4
                tickers = table['EPIC'].tolist()
                return [f"{t}.L" for t in tickers]
            elif index_name == "IBEX35":
                url = "https://en.wikipedia.org/wiki/IBEX_35"
                table = pd.read_html(url)[1]
                tickers = table['Ticker'].tolist()
                return [f"{t}.MC" for t in tickers]
        except Exception as e:
            st.error(f"Failed to fetch tickers: {e}")
            return []

    def add_indicators(self, df):
        """Adds technical indicators to a single dataframe."""
        if df.empty or len(df) < 20: return pd.DataFrame()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
        # MAs and Volatility
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=21).std()
        
        return df.dropna()