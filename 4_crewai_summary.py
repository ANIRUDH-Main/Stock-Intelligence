# =============================================================================
# 4_crewai_summary.py — CrewAI Narrative Summarizer
# Single CrewAI agent backed by local Ollama (llama3).
# Reads filtered_data.json, groups posts by ticker,
# produces a structured bull/bear narrative summary per ticker.
# Output: data/summaries.json
#
# Prerequisites:
#   1. Install Ollama: https://ollama.com
#   2. Pull model: ollama pull llama3
#   3. Ensure Ollama server is running: ollama serve
#
# Run: python scripts/4_crewai_summary.py
# =============================================================================

import json
import logging
import os
import sys
from datetime import datetime, timezone
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FILTERED_DATA_PATH, SUMMARIES_PATH,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    MAX_POSTS_PER_TICKER_FOR_SUMMARY
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
# LLM Setup
# -----------------------------------------------------------------------------

def init_llm():
    """
    Initialise local Ollama LLM via LangChain community integration.
    CrewAI uses LangChain under the hood, so this is the correct bridge.
    """
    log.info(f"Connecting to Ollama at {OLLAMA_BASE_URL} with model '{OLLAMA_MODEL}'...")
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,      # Low temperature = more factual, less creative hallucination
    )
    return llm

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------

def group_posts_by_ticker(filtered_data: dict) -> dict[str, list[dict]]:
    """
    Build a dict: ticker → list of relevant posts.
    A post can appear under multiple tickers if it mentions more than one.
    Caps list length at MAX_POSTS_PER_TICKER_FOR_SUMMARY to control token usage.
    """
    ticker_posts: dict[str, list[dict]] = {}

    for post in filtered_data.get("posts", []):
        for ticker in post.get("tickers", []):
            ticker_posts.setdefault(ticker, [])
            if len(ticker_posts[ticker]) < MAX_POSTS_PER_TICKER_FOR_SUMMARY:
                ticker_posts[ticker].append(post)

    return ticker_posts


def format_posts_for_prompt(ticker: str, posts: list[dict]) -> str:
    """
    Serialise posts into a compact text block for the LLM prompt.
    Strips noise: only includes title, body (truncated), and top comments.
    """
    lines = [f"== Reddit Discussion about {ticker} ==\n"]

    for i, post in enumerate(posts, 1):
        title   = post.get("title", "").strip()
        body    = (post.get("body", "") or "").strip()[:300]  # Truncate long bodies
        vader   = post.get("aggregate_vader", {})
        label   = post.get("sentiment_label", "neutral")
        score   = vader.get("compound", 0.0)
        subreddit = post.get("subreddit", "")

        lines.append(f"[Post {i}] r/{subreddit} | Sentiment: {label} ({score:+.2f})")
        lines.append(f"Title: {title}")
        if body:
            lines.append(f"Body: {body}...")

        # Include top 3 comments only
        top_comments = post.get("comments", [])[:3]
        for j, c in enumerate(top_comments, 1):
            c_text = (c.get("body", "") or "").strip()[:200]
            if c_text:
                lines.append(f"  Comment {j}: {c_text}")

        lines.append("")  # Blank line between posts

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# CrewAI Agent & Task
# -----------------------------------------------------------------------------

def create_summarizer_agent(llm) -> Agent:
    """
    Single agent: a Wall Street analyst persona.
    Role is intentionally specific — CrewAI performs better with focused roles.
    """
    return Agent(
        role="Senior Stock Market Analyst",
        goal=(
            "Analyse Reddit discussions about a stock and produce a concise, "
            "structured market narrative summary. Identify whether the community "
            "sentiment is bullish, bearish, or neutral, and explain WHY."
        ),
        backstory=(
            "You are a senior equity analyst who monitors social media for retail "
            "investor sentiment signals. You excel at distilling thousands of Reddit "
            "posts into clear, actionable summaries that capture key themes, catalysts, "
            "risks, and the overall market narrative without bias or hyperbole."
        ),
        llm=llm,
        verbose=False,      # Set True for debug output
        allow_delegation=False,  # Single agent — no delegation needed
        max_iter=3,         # Limit retries to control token usage
        max_execution_time=120
    )


def create_summary_task(agent: Agent, ticker: str, posts_text: str) -> Task:
    """
    Define the summarization task with a structured output format.
    The JSON format is specified explicitly so we can parse it reliably.
    """
    return Task(
        description=f"""
You have been given Reddit posts and comments discussing the stock {ticker}.

Here is the Reddit data:
{posts_text}

Your task is to analyse this data and produce a JSON response with the following structure:
{{
  "ticker": "{ticker}",
  "overall_verdict": "bullish" | "bearish" | "neutral",
  "confidence": "high" | "medium" | "low",
  "summary": "2-3 sentence plain English summary of the overall narrative",
  "bull_case": "Key reasons why Reddit is optimistic (or 'None mentioned' if absent)",
  "bear_case": "Key reasons why Reddit is pessimistic (or 'None mentioned' if absent)",
  "key_catalysts": ["catalyst 1", "catalyst 2", ...],
  "key_risks": ["risk 1", "risk 2", ...],
  "notable_themes": ["theme 1", "theme 2", ...],
  "post_count_analysed": {len(posts_text.split("[Post"))-1}
}}

IMPORTANT: 
- Respond ONLY with raw valid JSON.
- Do not use markdown.
- Do not explain anything.
- Your first character must be {{
- Your last character must be }}
- Base your analysis ONLY on the provided Reddit data. Do not use external knowledge.
- Be concise. Each field should be 1-2 sentences maximum.
""",
        agent=agent,
        expected_output=f"A valid JSON object summarising Reddit sentiment for {ticker}.",
    )

# -----------------------------------------------------------------------------
# Parse LLM output
# -----------------------------------------------------------------------------

def parse_llm_json(raw_output: str, ticker: str) -> dict:
    """
    Attempt to parse the LLM's response as JSON.
    LLMs sometimes wrap JSON in markdown fences — strip those first.
    Falls back to a structured error dict if parsing fails.
    """
    print(f"\n=== RAW OUTPUT FOR {ticker} ===\n{raw_output}\n=== END ===\n")
    # Strip markdown code fences if present
    cleaned = str(raw_output).strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        #print(cleaned[:1000])
        return json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning(f"  JSON parse failed for {ticker}. Storing raw output.")
        return {
            "ticker":           ticker,
            "overall_verdict":  "unknown",
            "confidence":       "low",
            "summary":          raw_output[:500],  # Store partial output for debugging
            "parse_error":      True,
        }

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("STEP 4 — CrewAI Narrative Summarizer")
    log.info("=" * 60)

    # Load filtered data
    if not os.path.exists(FILTERED_DATA_PATH):
        log.error("filtered_data.json not found. Run 2_sentiment.py first.")
        sys.exit(1)

    with open(FILTERED_DATA_PATH, "r", encoding="utf-8") as f:
        filtered_data = json.load(f)

    # Group posts by ticker
    ticker_posts = group_posts_by_ticker(filtered_data)
    log.info(f"Tickers to summarise: {sorted(ticker_posts.keys())}")

    if not ticker_posts:
        log.error("No tickers found. Exiting.")
        sys.exit(1)

    # Initialise LLM and agent
    try:
        llm = init_llm()
    except Exception as e:
        log.error(f"Failed to connect to Ollama: {e}")
        log.error("Make sure Ollama is running: `ollama serve`")
        sys.exit(1)

    agent = create_summarizer_agent(llm)

    # Process each ticker
    summaries = {}

    for ticker, posts in ticker_posts.items():
        log.info(f"\nSummarising {ticker} ({len(posts)} posts)...")

        posts_text = format_posts_for_prompt(ticker, posts)

        # Create a fresh task + crew per ticker
        # (CrewAI Crew is stateful — recreating ensures clean context per ticker)
        task = create_summary_task(agent, ticker, posts_text)

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            result = crew.kickoff()
            # CrewAI returns the final task output as a string
            raw_output = result if isinstance(result, str) else str(result)
            summary = parse_llm_json(raw_output, ticker)
        except Exception as e:
            log.error(f"  CrewAI failed for {ticker}: {e}")
            summary = {
                "ticker":          ticker,
                "overall_verdict": "unknown",
                "confidence":      "low",
                "summary":         f"Summary generation failed: {str(e)}",
                "error":           True,
            }

        summaries[ticker] = summary
        log.info(f"  ✅ {ticker}: {summary.get('overall_verdict', 'unknown').upper()} "
                 f"({summary.get('confidence', '?')} confidence)")

    # Save all summaries
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used":   OLLAMA_MODEL,
        "tickers":      sorted(summaries.keys()),
        "summaries":    summaries,
    }

    with open(SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"\n✅ All summaries saved → {SUMMARIES_PATH}")

    # Final summary log
    log.info("\n--- Verdict Summary ---")
    for ticker, s in sorted(summaries.items()):
        verdict    = s.get("overall_verdict", "unknown")
        confidence = s.get("confidence", "?")
        emoji = "🟢" if verdict == "bullish" else ("🔴" if verdict == "bearish" else "🟡")
        log.info(f"  {emoji} {ticker:<6}: {verdict.upper():<8} (confidence: {confidence})")


if __name__ == "__main__":
    main()
