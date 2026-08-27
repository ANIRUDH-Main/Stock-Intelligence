# Stock Intelligence & Sentiment Analysis Dashboard

An AI-powered financial intelligence pipeline that scrapes Reddit, detects stock tickers, scores sentiment, predicts next-day prices using ML, and generates analyst narratives via a local LLM — all displayed in an interactive Streamlit dashboard. **Zero ongoing API cost.**

---

## What It Does

| Stage | What Happens |
|-------|-------------|
| 1. Reddit Scraping | Collects posts & comments from 20 financial subreddits using PRAW |
| 2. Sentiment Analysis | Detects tickers via spaCy NER + regex, scores every post with VADER |
| 3. ML Prediction | Trains 5 regression models on 25 technical features per ticker |
| 4. AI Narratives | CrewAI + Ollama (llama3) generates structured analyst summaries locally |
| 5. Dashboard | Interactive Streamlit app with charts, sentiment gauges, and ML panels |

---

## Demo

> Streamlit dashboard showing candlestick chart, VADER sentiment gauge, ML prediction panel, and CrewAI analyst narrative for a selected ticker.

---

## Tech Stack

- **Data Collection** — PRAW (Reddit API), yfinance (Yahoo Finance)
- **NLP** — spaCy (`en_core_web_sm`), VADER Sentiment
- **Machine Learning** — XGBoost, Random Forest, Gradient Boosting, Ridge Regression, SVR (scikit-learn)
- **Agentic AI** — CrewAI, Ollama (llama3)
- **Dashboard** — Streamlit, Plotly
- **Core** — Python, Pandas, NumPy

---

## Project Structure

```
stock-sentiment-dashboard/
│
├── config.py                  # Reddit credentials, subreddit list, settings
├── requirements.txt
│
├── scripts/
│   ├── 1_scraper.py           # Stage 1: Reddit scraping
│   ├── 2_sentiment.py         # Stage 2: Ticker detection + VADER scoring
│   ├── 3_ml_predictor.py      # Stage 3: Feature engineering + model training
│   └── 4_crewai_summary.py    # Stage 4: CrewAI agent narratives
│
├── dashboard/
│   └── app.py                 # Stage 5: Streamlit dashboard
│
├── data/
│   ├── raw_data.json          # Output of Stage 1
│   ├── filtered_data.json     # Output of Stage 2
│   └── summaries.json         # Output of Stage 4
│
└── models/
    ├── model_metadata.json    # All ML metrics and predictions
    └── {TICKER}_best_model.pkl
```

---

## Subreddits Scraped

`r/wallstreetbets` `r/stocks` `r/investing` `r/StockMarket` `r/options` `r/pennystocks` `r/SecurityAnalysis` `r/ValueInvesting` `r/Daytrading` `r/algotrading` `r/GME` `r/Superstonk` `r/NVDA_Stock` `r/teslainvestorsclub` `r/apple` `r/technology` `r/dividends` `r/RobinHood` `r/finance` `r/economy`

---

## ML Models & Features

**5 models trained per ticker:**
- XGBoost Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression
- SVR (RBF kernel)

**25 engineered features per ticker:**
- Momentum: RSI(14), MACD, MACD Signal
- Trend: EMA(20), SMA(50)
- Volatility: Bollinger Bands (20-period, 2 SD)
- Volume: 20-day avg volume, volume ratio
- Lag prices: Close at 1, 2, 5 days ago
- Raw OHLCV data

**Performance targets:**
- R² > 0.85 on test set
- MAPE < 5% for large-cap tickers
- Chronological 80/20 train-test split (no data leakage)

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai) installed locally with llama3 pulled
- Reddit API credentials (free — takes 2 minutes at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps))

### 1. Clone the repo
```bash
git clone https://github.com/your-username/stock-sentiment-dashboard.git
cd stock-sentiment-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Add Reddit credentials
Open `config.py` and fill in:
```python
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "StockSentimentBot/1.0"
```

### 4. Pull the LLM (for Stage 4)
```bash
ollama pull llama3
```

---

## Running the Pipeline

Run each stage in order:

```bash
# Stage 1 — Scrape Reddit (2-5 min)
python scripts/1_scraper.py

# Stage 2 — Detect tickers + score sentiment (~30 sec)
python scripts/2_sentiment.py

# Stage 3 — Train ML models for every ticker found (5-15 min)
python scripts/3_ml_predictor.py

# Stage 4 — Generate CrewAI analyst narratives (2-10 min)
# Make sure Ollama is running before this step
python scripts/4_crewai_summary.py

# Stage 5 — Launch the dashboard
streamlit run dashboard/app.py
```

Then open `http://localhost:8501` in your browser.

> **Note:** Stages 1-4 only need to be re-run when you want fresh data. The dashboard reads from pre-generated JSON files and can be launched independently.

---

## Dashboard Features

- **Candlestick Chart** — Interactive Plotly price chart with EMA, SMA, and Bollinger Band overlays
- **Sentiment Panel** — VADER gauge showing bullish/bearish/neutral score per ticker, aggregated across all Reddit posts
- **ML Prediction Panel** — Next-day closing price prediction from the best-performing model, with RMSE and R² metrics
- **CrewAI Narrative** — Structured analyst summary: overall verdict, confidence, bull case, bear case, key catalysts, key risks
- **Reddit Post Explorer** — Browse raw posts filtered by sentiment label with expandable detail view

---

## Cost

| Resource | Cost |
|----------|------|
| Reddit API | Free |
| Yahoo Finance (yfinance) | Free |
| Ollama + llama3 | Free (runs locally) |
| All ML libraries | Free (open-source) |
| **Total ongoing cost** | **$0** |

---

## Authors

- **Anirudh Chaurasia** — [github.com/ANIRUDH-Main](https://github.com/ANIRUDH-Main) · [LinkedIn](https://linkedin.com/in/anirudh-chaurasia)
- **[Your Friend's Name]** — [GitHub](#) · [LinkedIn](#)

---

## License

MIT License — free to use, modify, and distribute.
