import pandas as pd
import numpy as np

def add_indicators(df):
    """
    Adds technical indicators to the DataFrame.
    Required columns: Close, Volume
    """
    if df.empty or len(df) < 200: # Need 200 for MA200
        return pd.DataFrame()
    
    df = df.copy()
    
    # Handle yfinance MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure we are using Series for calculations
    close = df['Close']
    volume = df['Volume']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['MA50'] = close.rolling(window=50).mean()
    df['MA200'] = close.rolling(window=200).mean()
    
    # Volatility (21-day rolling std of returns)
    df['Volatility'] = close.pct_change().rolling(window=21).std()
    
    # Volume Change (relative to 20-day average)
    volume_ma20 = volume.rolling(window=20).mean()
    df['Relative_Volume'] = volume / (volume_ma20 + 1e-9)
    
    return df.dropna()
