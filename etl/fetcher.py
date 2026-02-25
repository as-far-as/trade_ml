import pandas as pd
import requests
import streamlit as st

@st.cache_data(ttl=86400)
def get_index_tickers(index_name="SP500"):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        # Map indices to their Wikipedia table index
        mapping = {"SP500": (0, "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
                   "FTSE100": (4, "https://en.wikipedia.org/wiki/FTSE_100_Index"),
                   "IBEX35": (1, "https://en.wikipedia.org/wiki/IBEX_35")}
        
        table_idx, url = mapping[index_name]
        response = requests.get(url, headers=headers)
        df = pd.read_html(response.text, flavor='lxml')[table_idx]
        
        # SEARCH FOR TICKER COLUMN
        potential_cols = ['Symbol', 'Ticker', 'EPIC', 'Ticker symbol', 'Code']
        found_col = next((c for c in potential_cols if c in df.columns), None)
        
        if not found_col:
            st.error(f"Could not find ticker column in {index_name} table.")
            return []

        tickers = df[found_col].astype(str).str.strip().tolist()
        
        # APPLY SUFFIXES
        if index_name == "FTSE100": return [f"{t}.L" for t in tickers if not t.endswith('.L')]
        if index_name == "IBEX35": return [f"{t}.MC" for t in tickers if not t.endswith('.MC')]
        return [t.replace('.', '-') for t in tickers]

    except Exception as e:
        st.error(f"Scraping Error: {e}")
        return []