# =============================================================================
# config.py — Central configuration for Stock Intelligence Dashboard
# Edit this file to customize subreddits, tickers, paths, and model settings.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.json")
FILTERED_DATA_PATH = os.path.join(DATA_DIR, "filtered_data.json")
SUMMARIES_PATH = os.path.join(DATA_DIR, "summaries.json")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Auto-create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# REDDIT / PRAW SETTINGS
# -----------------------------------------------------------------------------
# Fill these in with your Reddit API credentials from https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID = "Mmld-sUj5f8by1kVHoP6rg"
REDDIT_CLIENT_SECRET = "4qR-KWsCOY9Z74qxLky9MXGwGT_09g"
REDDIT_USER_AGENT = "StockDashboard/1.0 by vibhor2"

# How many days back to scrape posts
SCRAPE_DAYS_BACK = 3

# Number of top posts to pull per subreddit (hot + new combined)
POSTS_PER_SUBREDDIT = 50

# Number of top comments to pull per post
COMMENTS_PER_POST = 20

# 20 subreddits — mix of general finance + stock-specific communities
SUBREDDITS = [
    "wallstreetbets",     # High-volume meme stocks, options plays
    "stocks",             # General stock discussion
    "investing",          # Long-term investing
    "StockMarket",        # Broad market news
    "options",            # Options trading
    "pennystocks",        # Small cap / penny stocks
    "SecurityAnalysis",   # Deep fundamental analysis
    "ValueInvesting",     # Buffett-style value investing
    "Daytrading",         # Short-term trades
    "algotrading",        # Algorithmic / quant trading
    "finance",            # Macro finance topics
    "economy",            # Economic news affecting markets
    "RobinHood",          # Retail investor community
    "Superstonk",         # GME deep dives
    "GME",                # GameStop specific
    "NVDA_Stock",         # NVIDIA specific
    "teslainvestorsclub", # Tesla investors
    "apple",              # Apple discussion
    "technology",         # Tech sector news
    "dividends",          # Dividend investing
]

# -----------------------------------------------------------------------------
# TICKER DETECTION SETTINGS
# -----------------------------------------------------------------------------

# Master ticker → company name mapping used for:
#   1. NER entity → ticker resolution (e.g. "Apple" → "AAPL")
#   2. Regex fallback matching bare tickers (e.g. "TSLA", "$TSLA")
# Add or remove tickers here to expand/contract the detection scope.
TICKER_ALIASES = {
    # Mega caps
    "AAPL":  ["Apple", "Apple Inc", "AAPL"],
    "MSFT":  ["Microsoft", "MSFT"],
    "GOOGL": ["Google", "Alphabet", "GOOGL", "GOOG"],
    "AMZN":  ["Amazon", "AMZN"],
    "NVDA":  ["Nvidia", "NVDA"],
    "META":  ["Meta", "Facebook", "META"],
    "TSLA":  ["Tesla", "TSLA"],
    "BRK":   ["Berkshire", "Berkshire Hathaway", "BRK"],
    "JPM":   ["JPMorgan", "JP Morgan", "JPM"],
    "V":     ["Visa", "V"],
    "MA":    ["Mastercard", "MA"],
    "UNH":   ["UnitedHealth", "United Health", "UNH"],
    "XOM":   ["ExxonMobil", "Exxon", "XOM"],
    "JNJ":   ["Johnson & Johnson", "Johnson Johnson", "JNJ"],
    "WMT":   ["Walmart", "WMT"],
    # Popular growth / tech
    "AMD":   ["AMD", "Advanced Micro Devices"],
    "INTC":  ["Intel", "INTC"],
    "NFLX":  ["Netflix", "NFLX"],
    "DIS":   ["Disney", "DIS"],
    "PYPL":  ["PayPal", "PYPL"],
    "SQ":    ["Block", "Square", "SQ"],
    "SHOP":  ["Shopify", "SHOP"],
    "SPOT":  ["Spotify", "SPOT"],
    "UBER":  ["Uber", "UBER"],
    "LYFT":  ["Lyft", "LYFT"],
    "SNAP":  ["Snap", "Snapchat", "SNAP"],
    "TWTR":  ["Twitter", "TWTR"],
    "COIN":  ["Coinbase", "COIN"],
    "HOOD":  ["Robinhood", "HOOD"],
    # Meme stocks
    "GME":   ["GameStop", "GME"],
    "AMC":   ["AMC", "AMC Entertainment"],
    "BB":    ["BlackBerry", "BB"],
    "BBBY":  ["Bed Bath Beyond", "BBBY"],
    # ETFs commonly discussed
    "SPY":   ["SPY", "S&P 500 ETF"],
    "QQQ":   ["QQQ", "Nasdaq ETF"],
    "ARKK":  ["ARKK", "ARK Innovation"],
}

# Minimum characters a ticker symbol must have to be matched by bare regex
# (avoids matching common words like "A", "I", "IT")
MIN_TICKER_LENGTH = 2

# -----------------------------------------------------------------------------
# SENTIMENT SETTINGS (VADER)
# -----------------------------------------------------------------------------

# Posts/comments with |compound| score below this are considered neutral noise
VADER_NEUTRAL_THRESHOLD = 0.05

# -----------------------------------------------------------------------------
# ML MODEL SETTINGS
# -----------------------------------------------------------------------------

# How many calendar days of yfinance history to download per ticker
YFINANCE_PERIOD = "1y"

# Fraction of data held out for testing (chronological split, not random)
TEST_SPLIT_RATIO = 0.2

# Technical indicator parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_PERIOD = 20
SMA_PERIOD = 50
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Models to train — all are regressors predicting next-day closing price
# Keys are used as display names in the dashboard and metadata JSON
MODELS_TO_TRAIN = [
    "XGBoost",
    "RandomForest",
    "Ridge",
    "SVR",
    "GradientBoosting",
]

# Minimum number of trading days required to train a model for a ticker
# Tickers with less data than this are skipped
MIN_TRADING_DAYS = 60

# -----------------------------------------------------------------------------
# CREWAI / OLLAMA SETTINGS
# -----------------------------------------------------------------------------

# Ollama model name — must be pulled locally before running:
#   ollama pull llama3
OLLAMA_MODEL = "llama3"

# Ollama base URL (default local install)
OLLAMA_BASE_URL = "http://localhost:11434"

# Maximum number of posts fed to CrewAI per ticker to control token usage
MAX_POSTS_PER_TICKER_FOR_SUMMARY = 1

# -----------------------------------------------------------------------------
# STREAMLIT DASHBOARD SETTINGS
# -----------------------------------------------------------------------------

# Number of days of price history shown on the candlestick chart
CHART_LOOKBACK_DAYS = 30

# Sentiment gauge color thresholds (VADER compound score)
SENTIMENT_BULLISH_THRESHOLD = 0.05
SENTIMENT_BEARISH_THRESHOLD = -0.05
