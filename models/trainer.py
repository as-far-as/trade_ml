import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def prepare_ml_data(df, target_horizon=5, buy_threshold=0.02):
    """
    Labels the data: 1 if price rises > 2% in 5 days, else 0.
    """
    df = df.copy()
    # Create the 'Target': Did the price go up X% in the next Y days?
    df['Future_Return'] = df['Close'].shift(-target_horizon) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > buy_threshold).astype(int)
    
    # Select features for the ML model
    features = ['RSI', 'MA50', 'MA200', 'Volatility', 'Relative_Volume', 'Sentiment']
    
    # If Sentiment isn't in df, fill it with 0 (neutral)
    if 'Sentiment' not in df.columns:
        df['Sentiment'] = 0.0
    
    # Drop rows with NaN values (created by indicators or shifting)
    ml_df = df[features + ['Target']].dropna()
    
    return ml_df[features], ml_df['Target']

def train_model(ticker_df, model_path='models/random_forest_model.pkl'):
    X, y = prepare_ml_data(ticker_df)
    
    if len(X) < 100:
        print("Not enough data to train model.")
        return
    
    # Split into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize and train the Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Check how well it performed
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
