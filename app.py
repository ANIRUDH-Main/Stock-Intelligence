# =============================================================================
# dashboard/app.py — Streamlit Dashboard
# Stock Intelligence & Social Sentiment Dashboard
#
# Run from project root: streamlit run dashboard/app.py
# =============================================================================

import json
import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FILTERED_DATA_PATH, MODEL_METADATA_PATH, SUMMARIES_PATH,
    CHART_LOOKBACK_DAYS,
    SENTIMENT_BULLISH_THRESHOLD, SENTIMENT_BEARISH_THRESHOLD,
    EMA_PERIOD, SMA_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD
)

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Stock Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    /* Metric card styling */
    div[data-testid="metric-container"] {
        background-color: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px 16px;
    }
    /* Verdict badge colors via custom classes */
    .badge-bullish  { background:#1a472a; color:#40d158; padding:4px 12px; border-radius:12px; font-weight:700; }
    .badge-bearish  { background:#4a1020; color:#ff6b6b; padding:4px 12px; border-radius:12px; font-weight:700; }
    .badge-neutral  { background:#2a2a1a; color:#ffd60a; padding:4px 12px; border-radius:12px; font-weight:700; }
    .badge-unknown  { background:#2a2a2a; color:#aaaaaa; padding:4px 12px; border-radius:12px; font-weight:700; }
    /* Section headers */
    .section-header { font-size:1.1rem; font-weight:600; color:#cdd6f4; margin-bottom:8px; }
    /* Post card */
    .post-card {
        background:#1e1e2e; border:1px solid #313244; border-radius:8px;
        padding:12px; margin-bottom:10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Data Loaders (cached)
# =============================================================================

@st.cache_data(ttl=300)  # Re-read files at most every 5 minutes
def load_filtered_data() -> dict:
    if not os.path.exists(FILTERED_DATA_PATH):
        return {}
    with open(FILTERED_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_model_metadata() -> dict:
    if not os.path.exists(MODEL_METADATA_PATH):
        return {}
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_summaries() -> dict:
    if not os.path.exists(SUMMARIES_PATH):
        return {}
    with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=900)  # Price data: re-fetch every 15 minutes
def fetch_price_history(ticker: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance for the candlestick chart."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 10)  # Extra buffer for weekends/holidays
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.tail(days)
    except Exception:
        return pd.DataFrame()

# =============================================================================
# Chart Builders
# =============================================================================

def build_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plotly candlestick chart with EMA, SMA, and Bollinger Bands overlaid."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No price data available", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(color="#cdd6f4", size=14))
        return fig

    close = df["Close"]

    # Compute overlays
    ema20 = close.ewm(span=EMA_PERIOD, adjust=False).mean()
    sma50 = close.rolling(window=min(SMA_PERIOD, len(df))).mean()
    bb_mid   = close.rolling(window=min(BOLLINGER_PERIOD, len(df))).mean()
    bb_std   = close.rolling(window=min(BOLLINGER_PERIOD, len(df))).std()
    bb_upper = bb_mid + BOLLINGER_STD * bb_std
    bb_lower = bb_mid - BOLLINGER_STD * bb_std

    fig = go.Figure()

    # Bollinger Band fill
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(bb_upper) + list(bb_lower[::-1]),
        fill="toself",
        fillcolor="rgba(100, 100, 255, 0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Bollinger Bands",
        hoverinfo="skip",
    ))

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name=ticker,
        increasing_line_color="#40d158",
        decreasing_line_color="#ff6b6b",
    ))

    # EMA 20
    fig.add_trace(go.Scatter(
        x=df.index, y=ema20,
        line=dict(color="#cba6f7", width=1.5),
        name=f"EMA {EMA_PERIOD}",
    ))

    # SMA 50
    fig.add_trace(go.Scatter(
        x=df.index, y=sma50,
        line=dict(color="#89dceb", width=1.5, dash="dot"),
        name=f"SMA {SMA_PERIOD}",
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Last {CHART_LOOKBACK_DAYS} Days", font=dict(color="#cdd6f4")),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#181825",
        plot_bgcolor="#181825",
        xaxis=dict(gridcolor="#313244", color="#cdd6f4"),
        yaxis=dict(gridcolor="#313244", color="#cdd6f4"),
        legend=dict(bgcolor="#1e1e2e", font=dict(color="#cdd6f4")),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    return fig


def build_sentiment_gauge(compound: float, ticker: str) -> go.Figure:
    """Plotly gauge chart for VADER compound score (-1 to +1)."""
    if compound >= SENTIMENT_BULLISH_THRESHOLD:
        color = "#40d158"
        label = "Bullish"
    elif compound <= SENTIMENT_BEARISH_THRESHOLD:
        color = "#ff6b6b"
        label = "Bearish"
    else:
        color = "#ffd60a"
        label = "Neutral"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(compound, 3),
        number=dict(font=dict(color=color, size=28)),
        title=dict(text=f"{ticker} VADER Sentiment<br><span style='font-size:13px'>{label}</span>",
                   font=dict(color="#cdd6f4")),
        gauge=dict(
            axis=dict(range=[-1, 1], tickcolor="#cdd6f4",
                      tickfont=dict(color="#cdd6f4"), nticks=5),
            bar=dict(color=color),
            bgcolor="#313244",
            borderwidth=0,
            steps=[
                dict(range=[-1, -SENTIMENT_BEARISH_THRESHOLD],   color="#4a1020"),
                dict(range=[-SENTIMENT_BEARISH_THRESHOLD, SENTIMENT_BULLISH_THRESHOLD], color="#2a2a1a"),
                dict(range=[SENTIMENT_BULLISH_THRESHOLD, 1],    color="#1a3a20"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.8,
                value=compound,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="#181825",
        font=dict(color="#cdd6f4"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=280,
    )
    return fig


def build_model_comparison_chart(all_metrics: dict) -> go.Figure:
    """Bar chart comparing RMSE across all trained models for the selected ticker."""
    if not all_metrics:
        return go.Figure()

    models = list(all_metrics.keys())
    rmse   = [all_metrics[m]["rmse"] for m in models]
    r2     = [all_metrics[m]["r2"]   for m in models]

    # Colour the best (lowest RMSE) differently
    min_rmse = min(rmse)
    colors   = ["#40d158" if v == min_rmse else "#6c7086" for v in rmse]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=models, y=rmse,
        name="RMSE (lower=better)",
        marker_color=colors,
        text=[f"{v:.2f}" for v in rmse],
        textposition="outside",
        textfont=dict(color="#cdd6f4"),
    ))

    fig.update_layout(
        title=dict(text="Model Comparison — RMSE", font=dict(color="#cdd6f4")),
        paper_bgcolor="#181825",
        plot_bgcolor="#181825",
        xaxis=dict(gridcolor="#313244", color="#cdd6f4"),
        yaxis=dict(gridcolor="#313244", color="#cdd6f4", title="RMSE ($)"),
        legend=dict(bgcolor="#1e1e2e", font=dict(color="#cdd6f4")),
        margin=dict(l=10, r=10, t=40, b=10),
        height=260,
        showlegend=False,
    )
    return fig


def build_post_volume_chart(ticker_posts: list[dict]) -> go.Figure:
    """Bar chart of post count per subreddit for the selected ticker."""
    if not ticker_posts:
        return go.Figure()

    sub_counts: dict[str, int] = {}
    for post in ticker_posts:
        sub = post.get("subreddit", "unknown")
        sub_counts[sub] = sub_counts.get(sub, 0) + 1

    subs   = sorted(sub_counts.keys(), key=lambda x: -sub_counts[x])[:10]
    counts = [sub_counts[s] for s in subs]

    fig = go.Figure(go.Bar(
        x=[f"r/{s}" for s in subs],
        y=counts,
        marker_color="#89b4fa",
        text=counts,
        textposition="outside",
        textfont=dict(color="#cdd6f4"),
    ))

    fig.update_layout(
        title=dict(text="Posts per Subreddit", font=dict(color="#cdd6f4")),
        paper_bgcolor="#181825",
        plot_bgcolor="#181825",
        xaxis=dict(gridcolor="#313244", color="#cdd6f4", tickangle=-30),
        yaxis=dict(gridcolor="#313244", color="#cdd6f4"),
        margin=dict(l=10, r=10, t=40, b=60),
        height=260,
        showlegend=False,
    )
    return fig

# =============================================================================
# UI Helper Functions
# =============================================================================

def verdict_badge(verdict: str) -> str:
    css_class = f"badge-{verdict}" if verdict in ("bullish", "bearish", "neutral") else "badge-unknown"
    emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(verdict, "⚪")
    return f'<span class="{css_class}">{emoji} {verdict.upper()}</span>'


def direction_delta(current: float, predicted: float) -> str:
    """Return a Streamlit delta string showing predicted change."""
    diff = predicted - current
    pct  = (diff / current) * 100 if current else 0
    return f"${diff:+.2f} ({pct:+.2f}%)"


def render_posts_table(posts: list[dict]):
    """Render Reddit posts for a ticker as expandable cards."""
    if not posts:
        st.info("No Reddit posts found for this ticker.")
        return

    for post in posts[:20]:  # Cap at 20 for performance
        compound  = post.get("aggregate_vader", {}).get("compound", 0)
        label     = post.get("sentiment_label", "neutral")
        subreddit = post.get("subreddit", "")
        title     = post.get("title", "No title")
        score     = post.get("score", 0)
        n_comments = post.get("num_comments", 0)
        tickers   = post.get("tickers", [])
        created   = post.get("created_dt", "")[:10]

        label_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(label, "⚪")

        with st.expander(f"{label_emoji} [{compound:+.3f}] {title[:90]}{'...' if len(title)>90 else ''}"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Subreddit",  f"r/{subreddit}")
            col2.metric("Upvotes",    f"{score:,}")
            col3.metric("Comments",   f"{n_comments:,}")
            col4.metric("Date",       created)

            st.markdown(f"**Tickers mentioned:** `{'` `'.join(tickers)}`")

            body = (post.get("body", "") or "").strip()
            if body:
                st.markdown("**Post body:**")
                st.markdown(f"> {body[:600]}{'...' if len(body)>600 else ''}")

            comments = post.get("comments", [])[:5]
            if comments:
                st.markdown("**Top comments:**")
                for c in comments:
                    c_text    = (c.get("body", "") or "").strip()[:300]
                    c_score   = c.get("score", 0)
                    c_compound = c.get("vader_scores", {}).get("compound", 0)
                    if c_text:
                        st.markdown(
                            f"- _{c_text}_ "
                            f"(👍 {c_score} | sentiment: {c_compound:+.3f})"
                        )

# =============================================================================
# Main Dashboard
# =============================================================================

def main():
    # -------------------------------------------------------------------
    # Load all data
    # -------------------------------------------------------------------
    filtered_data   = load_filtered_data()
    model_metadata  = load_model_metadata()
    summaries_data  = load_summaries()

    # -------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------
    st.markdown("## 📈 Stock Intelligence & Social Sentiment Dashboard")
    st.caption(f"Data last processed: {filtered_data.get('processed_at', 'N/A')[:19].replace('T',' ')} UTC")

    # -------------------------------------------------------------------
    # Check if data is available
    # -------------------------------------------------------------------
    if not filtered_data:
        st.error("⚠️ No data found. Please run the pipeline scripts first:")
        st.code("""
python scripts/1_scraper.py
python scripts/2_sentiment.py
python scripts/3_ml_predictor.py
python scripts/4_crewai_summary.py
        """)
        return

    available_tickers = filtered_data.get("unique_tickers", [])
    if not available_tickers:
        st.warning("No stock tickers were detected in the scraped Reddit data.")
        return

    # -------------------------------------------------------------------
    # Sidebar — Ticker Selector + Pipeline Status
    # -------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        selected_ticker = st.selectbox(
            "Select Ticker",
            options=sorted(available_tickers),
            index=0,
        )

        st.markdown("---")
        st.markdown("### 📊 Pipeline Status")

        def status_icon(path):
            return "✅" if os.path.exists(path) else "❌"

        st.markdown(f"{status_icon(FILTERED_DATA_PATH)} Scraper + Sentiment")
        st.markdown(f"{status_icon(MODEL_METADATA_PATH)} ML Models")
        st.markdown(f"{status_icon(SUMMARIES_PATH)} CrewAI Summaries")

        st.markdown("---")
        st.markdown("### 📋 Overview")
        total_posts    = filtered_data.get("total_filtered_posts", 0)
        total_tickers  = len(available_tickers)
        st.metric("Posts with stock mentions", total_posts)
        st.metric("Unique tickers detected",   total_tickers)

        if model_metadata:
            trained = len(model_metadata.get("tickers_trained", []))
            st.metric("Tickers with ML models", trained)

    # -------------------------------------------------------------------
    # Fetch data for selected ticker
    # -------------------------------------------------------------------
    ticker = selected_ticker

    # Sentiment data
    ticker_sentiment = filtered_data.get("ticker_sentiment_summary", {}).get(ticker, {})
    compound = ticker_sentiment.get("compound", 0.0)
    post_count = ticker_sentiment.get("post_count", 0)

    # ML prediction data
    predictions     = model_metadata.get("predictions", {})
    ticker_ml       = predictions.get(ticker, {})

    # CrewAI summary
    summaries       = summaries_data.get("summaries", {})
    ticker_summary  = summaries.get(ticker, {})

    # Posts for this ticker
    all_posts       = filtered_data.get("posts", [])
    ticker_posts    = [p for p in all_posts if ticker in p.get("tickers", [])]

    # Price history
    price_df        = fetch_price_history(ticker, CHART_LOOKBACK_DAYS)

    # -------------------------------------------------------------------
    # Row 1: Key Metrics
    # -------------------------------------------------------------------
    st.markdown(f"### {ticker}")

    col1, col2, col3, col4, col5 = st.columns(5)

    # Current price
    current_price = ticker_ml.get("current_price")
    if current_price is None and not price_df.empty:
        current_price = float(price_df["Close"].iloc[-1])

    col1.metric(
        "Current Price",
        f"${current_price:.2f}" if current_price else "N/A",
    )

    # Predicted price
    predicted_price = ticker_ml.get("predicted_next_close")
    if predicted_price and current_price:
        col2.metric(
            "Predicted Tomorrow",
            f"${predicted_price:.2f}",
            delta=direction_delta(current_price, predicted_price),
        )
    else:
        col2.metric("Predicted Tomorrow", "N/A")

    # Best model
    col3.metric(
        "Best ML Model",
        ticker_ml.get("best_model", "N/A"),
        delta=f"R²={ticker_ml.get('best_model_metrics', {}).get('r2', 'N/A')}" if ticker_ml else None,
    )

    # VADER score
    col4.metric(
        "VADER Compound",
        f"{compound:+.4f}",
        delta=ticker_sentiment.get("label", "neutral"),
    )

    # Reddit post count
    col5.metric("Reddit Posts", post_count)

    st.markdown("---")

    # -------------------------------------------------------------------
    # Row 2: Price Chart + Sentiment Gauge
    # -------------------------------------------------------------------
    chart_col, gauge_col = st.columns([2.2, 1])

    with chart_col:
        st.markdown('<p class="section-header">📉 Price Chart</p>', unsafe_allow_html=True)
        fig_candle = build_candlestick_chart(price_df, ticker)
        st.plotly_chart(fig_candle, use_container_width=True)

    with gauge_col:
        st.markdown('<p class="section-header">🎯 Social Sentiment</p>', unsafe_allow_html=True)
        fig_gauge = build_sentiment_gauge(compound, ticker)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Post volume chart below gauge
        fig_volume = build_post_volume_chart(ticker_posts)
        st.plotly_chart(fig_volume, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------------
    # Row 3: ML Model Comparison + CrewAI Summary
    # -------------------------------------------------------------------
    ml_col, summary_col = st.columns([1, 1])

    with ml_col:
        st.markdown('<p class="section-header">🤖 ML Model Comparison</p>', unsafe_allow_html=True)

        if ticker_ml:
            all_metrics = ticker_ml.get("all_model_metrics", {})
            fig_models  = build_model_comparison_chart(all_metrics)
            st.plotly_chart(fig_models, use_container_width=True)

            # Metrics table
            if all_metrics:
                rows = []
                best = ticker_ml.get("best_model", "")
                for model_name, metrics in all_metrics.items():
                    rows.append({
                        "Model":  f"⭐ {model_name}" if model_name == best else model_name,
                        "RMSE":   f"${metrics['rmse']:.2f}",
                        "MAE":    f"${metrics['mae']:.2f}",
                        "R²":     f"{metrics['r2']:.4f}",
                        "MAPE %": f"{metrics['mape']:.2f}%",
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info(f"No ML model available for {ticker}. Run 3_ml_predictor.py.")

    with summary_col:
        st.markdown('<p class="section-header">🧠 CrewAI Reddit Narrative</p>', unsafe_allow_html=True)

        if ticker_summary and not ticker_summary.get("error") and not ticker_summary.get("parse_error"):
            verdict    = ticker_summary.get("overall_verdict", "unknown")
            confidence = ticker_summary.get("confidence", "?")

            # Verdict badge
            st.markdown(
                f'Verdict: {verdict_badge(verdict)} &nbsp; Confidence: <b>{confidence}</b>',
                unsafe_allow_html=True
            )
            st.markdown("")

            # Summary
            st.markdown("**📝 Summary**")
            st.markdown(ticker_summary.get("summary", "N/A"))

            # Bull / Bear case
            bcol, rcol = st.columns(2)
            with bcol:
                st.markdown("**🟢 Bull Case**")
                st.markdown(ticker_summary.get("bull_case", "None mentioned"))
            with rcol:
                st.markdown("**🔴 Bear Case**")
                st.markdown(ticker_summary.get("bear_case", "None mentioned"))

            # Catalysts + Risks
            catalysts = ticker_summary.get("key_catalysts", [])
            risks     = ticker_summary.get("key_risks", [])
            themes    = ticker_summary.get("notable_themes", [])

            if catalysts:
                st.markdown("**⚡ Key Catalysts**")
                st.markdown("\n".join([f"- {c}" for c in catalysts]))

            if risks:
                st.markdown("**⚠️ Key Risks**")
                st.markdown("\n".join([f"- {r}" for r in risks]))

            if themes:
                st.markdown("**💬 Notable Themes**")
                st.markdown("  ".join([f"`{t}`" for t in themes]))

        elif ticker_summary.get("parse_error"):
            st.warning("Summary generated but could not be parsed as JSON. Raw output:")
            st.text(ticker_summary.get("summary", ""))
        else:
            st.info(f"No CrewAI summary for {ticker}. Run 4_crewai_summary.py.")

    st.markdown("---")

    # -------------------------------------------------------------------
    # Row 4: Reddit Posts Explorer
    # -------------------------------------------------------------------
    st.markdown(f'<p class="section-header">📰 Reddit Posts mentioning {ticker} ({len(ticker_posts)} found)</p>',
                unsafe_allow_html=True)

    # Filter controls
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            ["All", "bullish", "neutral", "bearish"],
        )

    filtered_ticker_posts = ticker_posts
    if sentiment_filter != "All":
        filtered_ticker_posts = [p for p in ticker_posts if p.get("sentiment_label") == sentiment_filter]

    render_posts_table(filtered_ticker_posts)


if __name__ == "__main__":
    main()
