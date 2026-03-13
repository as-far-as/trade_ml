from google import genai
import streamlit as st
import json
import re

class SentimentAnalyzer:
    def __init__(self, model="gemini-1.5-flash"):
        self.model_name = model

    def analyze_sentiment(self, texts):
        """
        Analyzes a list of texts and returns a consolidated sentiment score between -1 and 1.
        """
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not texts or not api_key:
            return 0.0
        
        combined_text = "\n---\n".join(texts[:20]) # Limit to last 20 posts for context
        
        prompt = f"""
        Analyze the sentiment of the following news/posts about a stock. 
        Provide a single numerical sentiment score between -1 (extremely bearish) and 1 (extremely bullish), 
        where 0 is neutral. 
        
        Text:
        {combined_text}
        
        Output only the number.
        """
        
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            score_str = response.text.strip()
            
            # Basic validation to extract just the number if LLM gets wordy
            match = re.search(r"[-+]?\d*\.?\d+", score_str)
            if match:
                return float(match.group())
            return 0.0
        except Exception as e:
            st.error(f"Sentiment analysis failed: {e}")
            return 0.0
