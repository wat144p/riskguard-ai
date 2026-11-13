from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()


# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Tickers to track
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]

# Model settings
CLUSTERS_K = 3
PCA_COMPONENTS = 3

# Data settings
HISTORY_YEARS = 5         # price history to fetch
TEST_SIZE_DAYS = 90       # the last 90 trading dates are the held-out test window
LABEL_HORIZON_DAYS = 5    # "risk" = volatility of the next 5 daily returns
WALK_FORWARD_WINDOWS = 5  # how many consecutive 90-day windows the evaluation walks through

# Risk classes: 0 = Low, 1 = Medium, 2 = High
RISK_LEVELS = ["Low", "Medium", "High"]
