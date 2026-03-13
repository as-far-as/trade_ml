from etl.fetcher import DataFetcher
from models.trainer import train_model
import pandas as pd
import yfinance as yf

# 1. Fetch data for several diverse and more volatile stocks to get better training examples
tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOG", "AMD", "TSLA", "NFLX", "JPM"]
all_data = []

for t in tickers:
    print(f"Fetching data for {t}...")
    fetcher = DataFetcher(t)
    df = fetcher.fetch_historical_data(period="10y")
    if not df.empty:
        df_ind = fetcher.add_indicators(df)
        if not df_ind.empty:
            all_data.append(df_ind)

if all_data:
    combined_df = pd.concat(all_data)
    print(f"Training on {len(combined_df)} rows...")
    train_model(combined_df)
else:
    print("No data collected for training.")
