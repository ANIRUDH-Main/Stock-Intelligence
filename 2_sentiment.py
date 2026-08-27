# =============================================================================
# 2_sentiment.py — Sentiment Engine
# Phase 1: spaCy NER detects company names → maps to tickers
# Phase 2: Regex fallback for $TICKER / TICKER patterns
# Phase 3: VADER scores each post+comments block
# Output: data/filtered_data.json (only records with confirmed ticker mentions)
#
# Run: python scripts/2_sentiment.py
# =============================================================================

import json
import re
import logging
from datetime import datetime, timezone

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    TICKER_ALIASES, MIN_TICKER_LENGTH,
    VADER_NEUTRAL_THRESHOLD,
    RAW_DATA_PATH, FILTERED_DATA_PATH
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Build lookup structures from TICKER_ALIASES config
# -----------------------------------------------------------------------------

def build_lookup_structures():
    """
    From TICKER_ALIASES, build:
    1. alias_to_ticker: {"Tesla": "TSLA", "Apple": "AAPL", ...}
       Used by NER entity → ticker resolution.
    2. all_tickers: set of all valid ticker symbols.
       Used by regex fallback.
    3. sorted_aliases: list of (alias, ticker) sorted by alias length descending
       to prevent short aliases shadowing longer ones.
    """
    alias_to_ticker = {}
    all_tickers = set()

    for ticker, aliases in TICKER_ALIASES.items():
        all_tickers.add(ticker)
        for alias in aliases:
            # Lower-case key for case-insensitive matching
            alias_to_ticker[alias.lower()] = ticker

    # Sort by alias length desc so "Apple Inc" matches before "Apple"
    sorted_aliases = sorted(alias_to_ticker.items(), key=lambda x: len(x[0]), reverse=True)

    return alias_to_ticker, all_tickers, sorted_aliases


ALIAS_TO_TICKER, ALL_TICKERS, SORTED_ALIASES = build_lookup_structures()

# Regex for explicit ticker patterns: $TSLA, TSLA (all caps 2-5 chars)
# Word boundary anchors prevent matching inside other words (e.g. "GAMESTOP")
TICKER_REGEX = re.compile(
    r'\$([A-Z]{' + str(MIN_TICKER_LENGTH) + r',5})'   # $TSLA style
    r'|(?<!\w)([A-Z]{' + str(MIN_TICKER_LENGTH) + r',5})(?!\w)',  # bare TSLA style
    re.UNICODE
)

# Stopwords: common all-caps words that are NOT tickers
# Without this, words like "CEO", "IPO", "ALL", "FOR" would be false positives
CAPS_STOPWORDS = {
    "I", "A", "THE", "AND", "OR", "FOR", "IN", "ON", "AT", "TO", "OF",
    "IS", "IT", "BY", "BE", "DO", "GO", "AN", "MY", "WE", "US", "HE",
    "SHE", "SO", "IF", "BUT", "NOT", "ARE", "HAS", "HAD", "WAS", "CAN",
    "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "USD", "EUR", "GDP", "CPI",
    "FED", "SEC", "NYSE", "NASDAQ", "EPS", "PE", "ATH", "ATL", "YOLO",
    "IMO", "IMHO", "DD", "OG", "IRL", "TBH", "LOL", "LMAO", "TL", "DR",
    "WSB", "GME", "AMC", "USA", "UK", "EU", "AI", "ML", "IT", "HR",
    "PR", "ROI", "YOY", "QOQ", "EOD", "EOY", "FYI", "FWIW", "AH", "PM",
    "AM", "OP", "OC", "TV", "PC", "OS", "API", "SaaS", "IPO", "ICO",
}

# -----------------------------------------------------------------------------
# NLP model setup
# -----------------------------------------------------------------------------

def load_spacy():
    """Load spaCy model. Falls back gracefully if not installed."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        log.info("spaCy en_core_web_sm loaded ✓")
        return nlp
    except OSError:
        log.warning(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm\n"
            "Falling back to regex-only detection."
        )
        return None


# -----------------------------------------------------------------------------
# Ticker Detection
# -----------------------------------------------------------------------------

def detect_tickers_ner(text: str, nlp) -> set[str]:
    """
    Phase 1: Use spaCy NER to find ORG entities in text,
    then map them to tickers via ALIAS_TO_TICKER.
    Returns a set of matched ticker symbols.
    """
    tickers = set()
    if nlp is None or not text.strip():
        return tickers

    doc = nlp(text[:100000])  # spaCy has a default max length; trim just in case

    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT"):
            entity_lower = ent.text.lower().strip()
            # Direct lookup
            if entity_lower in ALIAS_TO_TICKER:
                tickers.add(ALIAS_TO_TICKER[entity_lower])
                continue
            # Partial match: check if any alias is contained in the entity text
            for alias, ticker in SORTED_ALIASES:
                if alias in entity_lower:
                    tickers.add(ticker)
                    break

    return tickers


def detect_tickers_regex(text: str) -> set[str]:
    """
    Phase 2: Regex fallback.
    Matches $TSLA style and bare TSLA style (all-caps, 2-5 chars).
    Filters against ALL_TICKERS to only keep known symbols.
    Removes stopwords to cut false positives.
    """
    tickers = set()
    if not text.strip():
        return tickers

    for match in TICKER_REGEX.finditer(text):
        # Group 1 = $TICKER, group 2 = bare TICKER
        symbol = (match.group(1) or match.group(2) or "").upper()
        if not symbol:
            continue
        if symbol in CAPS_STOPWORDS:
            continue
        if symbol in ALL_TICKERS:
            tickers.add(symbol)

    return tickers


def detect_tickers(text: str, nlp) -> set[str]:
    """
    Combined detection: NER first, regex as fallback/supplement.
    Returns union of both sets.
    """
    ner_tickers = detect_tickers_ner(text, nlp)
    regex_tickers = detect_tickers_regex(text)
    return ner_tickers | regex_tickers


# -----------------------------------------------------------------------------
# VADER Sentiment
# -----------------------------------------------------------------------------

def score_text(analyzer: SentimentIntensityAnalyzer, text: str) -> dict:
    """Return VADER scores for a piece of text."""
    if not text or not text.strip():
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
    return analyzer.polarity_scores(text)


def aggregate_sentiment(post_score: dict, comment_scores: list[dict]) -> dict:
    """
    Compute a weighted aggregate VADER score for a post.
    Post title+body gets weight 2x vs individual comments.
    Returns the same dict structure: compound, pos, neu, neg.
    """
    all_scores = [post_score, post_score]  # Weight post 2x
    all_scores.extend(comment_scores)

    if not all_scores:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    n = len(all_scores)
    return {
        "compound": round(sum(s["compound"] for s in all_scores) / n, 4),
        "pos":      round(sum(s["pos"]      for s in all_scores) / n, 4),
        "neu":      round(sum(s["neu"]      for s in all_scores) / n, 4),
        "neg":      round(sum(s["neg"]      for s in all_scores) / n, 4),
    }


def sentiment_label(compound: float) -> str:
    """Convert compound score to human-readable label."""
    if compound >= VADER_NEUTRAL_THRESHOLD:
        return "bullish"
    elif compound <= -VADER_NEUTRAL_THRESHOLD:
        return "bearish"
    return "neutral"


# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------

def process_post(post: dict, nlp, analyzer: SentimentIntensityAnalyzer) -> dict | None:
    """
    Process a single post:
    1. Concatenate title + body for full text
    2. Run ticker detection on title + body + all comments
    3. Score sentiment on title + body block and each comment
    4. Return enriched post dict, or None if no tickers found
    """
    title = post.get("title", "") or ""
    body  = post.get("body", "")  or ""
    full_text = f"{title}\n{body}".strip()

    # Ticker detection — scan the full post text
    post_tickers = detect_tickers(full_text, nlp)

    # Also scan comments for additional ticker mentions
    comment_tickers = set()
    comment_scores  = []
    enriched_comments = []

    for comment in post.get("comments", []):
        c_text = comment.get("body", "") or ""
        c_tickers = detect_tickers(c_text, nlp)
        comment_tickers |= c_tickers
        c_score = score_text(analyzer, c_text)
        enriched_comments.append({
            **comment,
            "tickers":     list(c_tickers),
            "vader_scores": c_score,
        })
        comment_scores.append(c_score)

    all_tickers = post_tickers | comment_tickers

    # If no tickers found anywhere in this post, exclude from filtered output
    if not all_tickers:
        return None

    # Score the post itself
    post_score = score_text(analyzer, full_text)
    agg_score  = aggregate_sentiment(post_score, comment_scores)

    return {
        **post,
        "tickers":            sorted(all_tickers),
        "vader_scores":       post_score,
        "aggregate_vader":    agg_score,
        "sentiment_label":    sentiment_label(agg_score["compound"]),
        "comments":           enriched_comments,
    }


def build_ticker_summary(filtered_posts: list[dict]) -> dict:
    """
    Build a per-ticker aggregated sentiment summary across all filtered posts.
    Used by the dashboard for the sentiment gauge.
    Structure: { "TSLA": { "compound": 0.12, "label": "bullish", "post_count": 42 }, ... }
    """
    ticker_data: dict[str, list[float]] = {}

    for post in filtered_posts:
        compound = post["aggregate_vader"]["compound"]
        for ticker in post["tickers"]:
            ticker_data.setdefault(ticker, []).append(compound)

    summary = {}
    for ticker, scores in ticker_data.items():
        avg = round(sum(scores) / len(scores), 4)
        summary[ticker] = {
            "compound":   avg,
            "label":      sentiment_label(avg),
            "post_count": len(scores),
            "scores":     scores,  # Keep raw list for distribution plots in dashboard
        }

    return summary


def main():
    log.info("=" * 60)
    log.info("STEP 2 — Sentiment Engine")
    log.info("=" * 60)

    # Load raw data
    if not os.path.exists(RAW_DATA_PATH):
        log.error(f"raw_data.json not found at {RAW_DATA_PATH}. Run 1_scraper.py first.")
        sys.exit(1)

    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    posts = raw.get("posts", [])
    log.info(f"Loaded {len(posts)} posts from raw_data.json")

    # Initialise NLP tools
    nlp      = load_spacy()
    analyzer = SentimentIntensityAnalyzer()
    log.info("VADER sentiment analyser loaded ✓")

    # Process all posts
    filtered_posts = []
    skipped = 0

    for post in tqdm(posts, desc="Processing posts", unit="post"):
        result = process_post(post, nlp, analyzer)
        if result is not None:
            filtered_posts.append(result)
        else:
            skipped += 1

    log.info(f"Posts with stock mentions : {len(filtered_posts)}")
    log.info(f"Posts without mentions (skipped): {skipped}")

    # Build per-ticker sentiment summary
    ticker_summary = build_ticker_summary(filtered_posts)
    log.info(f"Unique tickers detected: {sorted(ticker_summary.keys())}")

    # Save filtered output
    output = {
        "processed_at":  datetime.now(timezone.utc).isoformat(),
        "source_file":   RAW_DATA_PATH,
        "total_filtered_posts": len(filtered_posts),
        "unique_tickers": sorted(ticker_summary.keys()),
        "ticker_sentiment_summary": ticker_summary,
        "posts": filtered_posts,
    }

    with open(FILTERED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"✅ Saved {len(filtered_posts)} filtered posts → {FILTERED_DATA_PATH}")

    # Quick summary table
    log.info("\n--- Ticker Sentiment Summary ---")
    for ticker, data in sorted(ticker_summary.items(), key=lambda x: -x[1]["post_count"]):
        log.info(
            f"  {ticker:<6} | {data['label']:<8} | "
            f"compound={data['compound']:+.4f} | posts={data['post_count']}"
        )


if __name__ == "__main__":
    main()
