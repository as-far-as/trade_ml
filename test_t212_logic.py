
import pandas as pd

def map_t212_to_yahoo(t212_ticker):
    """Maps Trading 212 ticker format to Yahoo Finance tickers."""
    parts = t212_ticker.split('_')
    symbol = parts[0]
    if len(parts) < 2: return symbol
    
    exchange = parts[1]
    mapping = {
        "L": ".L",    # London
        "ES": ".MC",  # Madrid
        "DE": ".DE",  # XETRA
        "FR": ".PA",  # Paris
        "AS": ".AS",  # Amsterdam
        "MI": ".MI",  # Milan
        "BR": ".BR",  # Brussels
        "HE": ".HE",  # Helsinki
        "ST": ".ST",  # Stockholm
        "OS": ".OL",  # Oslo
        "CO": ".CO",  # Copenhagen
        "SW": ".SW",  # Switzerland
        "US": ""      # US (No suffix)
    }
    return symbol + mapping.get(exchange, "")

def test_t212_mapping():
    # Test cases: (T212 Ticker, Expected Yahoo Ticker)
    test_cases = [
        ("TSLA_US_EQ", "TSLA"),
        ("BARC_L_EQ", "BARC.L"),
        ("ITX_ES_EQ", "ITX.MC"),
        ("SAP_DE_EQ", "SAP.DE"),
        ("AIR_FR_EQ", "AIR.PA"),
        ("AAPL", "AAPL"), # Fallback
    ]
    
    print("Testing T212 Ticker Mapping...")
    for t212, expected in test_cases:
        actual = map_t212_to_yahoo(t212)
        status = "✅" if actual == expected else "❌"
        print(f"[{status}] {t212} -> {actual} (Expected: {expected})")

def test_dataframe_processing():
    # Mock T212 API Response
    mock_pos = [
        {
            "ticker": "TSLA_US_EQ",
            "quantity": 10.0,
            "averagePrice": 200.0,
            "currentPrice": 210.0,
            "ppl": 100.0
        },
        {
            "ticker": "BARC_L_EQ",
            "quantity": 500.0,
            "averagePrice": 2.10,
            "currentPrice": 2.20,
            "ppl": 50.0
        }
    ]
    
    print("\nTesting Portfolio DataFrame Processing...")
    rows = []
    for p in mock_pos:
        t = map_t212_to_yahoo(p['ticker'])
        rows.append({
            "Ticker": t, 
            "Qty": p['quantity'], 
            "Avg Price": p['averagePrice'], 
            "Current Price": p['currentPrice'], 
            "P&L": p['ppl']
        })
    
    df = pd.DataFrame(rows)
    print("Processed DataFrame:")
    print(df)
    
    if "TSLA" in df['Ticker'].values and "BARC.L" in df['Ticker'].values:
        print("✅ DataFrame contains expected mapped tickers.")
    else:
        print("❌ DataFrame ticker mapping failed.")

if __name__ == "__main__":
    test_t212_mapping()
    test_dataframe_processing()
