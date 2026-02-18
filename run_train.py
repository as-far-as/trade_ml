from etl.fetcher import DataFetcher
from models.trainer import train_model
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Fetch data for a reliable "baseline" stock (like SPY or AAPL)
fetcher = DataFetcher("SPY")
raw_data = fetcher.fetch_historical_data(period="5y") # 5 years for better learning
df_with_indicators = fetcher.add_indicators(raw_data)

# 2. Run the training function from your trainer.py
train_model(df_with_indicators)