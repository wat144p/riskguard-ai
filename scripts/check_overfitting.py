"""Is the return regressor overfitting, or is there simply no signal?

Usage (from the repo root, after a training run or with the committed release model):
    python scripts/check_overfitting.py
"""
import sys
from pathlib import Path
from sklearn.metrics import r2_score

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from ml.config import TICKERS
from ml.data_ingestion import fetch_stock_data
from ml.feature_engineering import create_features, split_data
from ml.models import load_latest_models

def check_overfitting():
    print("Loading data...")
    # 1. Fetch and prepare data exactly as the training flow does
    df = fetch_stock_data(TICKERS, use_cache=True)
    df_features = create_features(df)
    train_df, test_df, _ = split_data(df_features)
    
    # 2. Load Models
    print("Loading latest models...")
    models = load_latest_models()
    regressor = models["regressor"]
    features = models["features"]
    
    # 3. Prepare X/y
    X_train = train_df[features]
    y_train = train_df["target_return_next_day"]
    
    X_test = test_df[features]
    y_test = test_df["target_return_next_day"]
    
    # 4. Predict
    train_preds = regressor.predict(X_train)
    test_preds = regressor.predict(X_test)
    
    # 5. Score
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print("\n" + "="*30)
    print(f"REGRESSION MODEL DIAGNOSTICS")
    print("="*30)
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2:  {test_r2:.4f}")
    print("-" * 30)
    
    if train_r2 > 0.5 and test_r2 < 0:
        print("DIAGNOSIS: SEVERE OVERFITTING")
        print("The model memorized the training data but fails on new data.")
    elif train_r2 > 0.1 and test_r2 < 0:
        print("DIAGNOSIS: Mild Overfitting")
    elif abs(train_r2) < 0.1 and abs(test_r2) < 0.1:
        print("DIAGNOSIS: Underfitting / No Signal")
        print("The model is effectively guessing the average. This is common in financial time series with little data.")
    else:
        print("DIAGNOSIS: Inconclusive / Other")
        
if __name__ == "__main__":
    check_overfitting()


