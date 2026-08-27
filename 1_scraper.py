# =============================================================================
# 1_scraper.py — Reddit Scraper
# Pulls posts + comments from configured subreddits using PRAW.
# Output: data/raw_data.json
#
# Run: python scripts/1_scraper.py
# =============================================================================

import json
import time
import logging
from datetime import datetime, timezone, timedelta

import praw
from tqdm import tqdm

# Import central config (works whether run from project root or scripts/)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    SUBREDDITS, POSTS_PER_SUBREDDIT, COMMENTS_PER_POST,
    SCRAPE_DAYS_BACK, RAW_DATA_PATH
)

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def init_reddit() -> praw.Reddit:
    """Initialise and return an authenticated PRAW Reddit instance."""
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    # Read-only mode — no login required for public posts
    reddit.read_only = True
    return reddit


def utc_cutoff() -> float:
    """Return UTC timestamp for SCRAPE_DAYS_BACK days ago."""
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=SCRAPE_DAYS_BACK)
    return cutoff_dt.timestamp()


def extract_post(submission) -> dict:
    """Convert a PRAW Submission object into a clean dictionary."""
    return {
        "post_id":    submission.id,
        "subreddit":  submission.subreddit.display_name,
        "title":      submission.title,
        "body":       submission.selftext,          # Empty string for link posts
        "url":        submission.url,
        "score":      submission.score,             # Net upvotes
        "upvote_ratio": submission.upvote_ratio,
        "num_comments": submission.num_comments,
        "created_utc": submission.created_utc,
        "created_dt": datetime.fromtimestamp(
            submission.created_utc, tz=timezone.utc
        ).isoformat(),
        "comments":   [],                           # Filled below
    }


def extract_comments(submission, cutoff_ts: float) -> list[dict]:
    """
    Pull top-level comments from a submission.
    Replaces MoreComments objects so we get actual text.
    Skips deleted/removed comments and comments older than cutoff.
    """
    comments = []
    try:
        # replace_more(limit=0) skips loading "load more comments" chains
        # to keep API calls low. Use limit=5 for deeper threads if needed.
        submission.comments.replace_more(limit=0)
    except Exception as e:
        log.warning(f"replace_more failed for {submission.id}: {e}")
        return comments

    for comment in submission.comments.list()[:COMMENTS_PER_POST]:
        # Skip deleted / removed
        if not comment.body or comment.body in ("[deleted]", "[removed]"):
            continue
        # Skip comments outside our date window
        if comment.created_utc < cutoff_ts:
            continue
        comments.append({
            "comment_id":   comment.id,
            "body":         comment.body,
            "score":        comment.score,
            "created_utc":  comment.created_utc,
            "created_dt":   datetime.fromtimestamp(
                comment.created_utc, tz=timezone.utc
            ).isoformat(),
        })
    return comments


def scrape_subreddit(reddit: praw.Reddit, subreddit_name: str, cutoff_ts: float) -> list[dict]:
    """
    Pull posts from both 'hot' and 'new' feeds of a subreddit.
    Deduplicates by post_id. Filters to posts within our date window.
    """
    posts = {}
    subreddit = reddit.subreddit(subreddit_name)

    feeds = [
        ("hot",  subreddit.hot(limit=POSTS_PER_SUBREDDIT)),
        ("new",  subreddit.new(limit=POSTS_PER_SUBREDDIT)),
    ]

    for feed_name, feed in feeds:
        try:
            for submission in feed:
                # Skip posts outside our time window
                if submission.created_utc < cutoff_ts:
                    continue
                # Skip stickied mod posts (announcements, not market discussion)
                if submission.stickied:
                    continue
                if submission.id not in posts:
                    post = extract_post(submission)
                    post["comments"] = extract_comments(submission, cutoff_ts)
                    posts[submission.id] = post
        except Exception as e:
            log.warning(f"Error on r/{subreddit_name} [{feed_name}]: {e}")
            # Small back-off on rate limit errors
            time.sleep(2)

    return list(posts.values())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("STEP 1 — Reddit Scraper")
    log.info(f"Scraping {len(SUBREDDITS)} subreddits, last {SCRAPE_DAYS_BACK} days")
    log.info("=" * 60)

    reddit = init_reddit()
    cutoff_ts = utc_cutoff()
    cutoff_dt = datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info(f"Cutoff: {cutoff_dt}")

    all_posts = []
    failed_subreddits = []

    for subreddit_name in tqdm(SUBREDDITS, desc="Subreddits", unit="sub"):
        try:
            posts = scrape_subreddit(reddit, subreddit_name, cutoff_ts)
            all_posts.extend(posts)
            log.info(f"r/{subreddit_name:<25} → {len(posts):>3} posts")
        except Exception as e:
            log.error(f"r/{subreddit_name} failed entirely: {e}")
            failed_subreddits.append(subreddit_name)
        # Polite delay to stay within Reddit's rate limit (~60 req/min)
        time.sleep(0.5)

    # Build the final output structure
    output = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "scrape_window_days": SCRAPE_DAYS_BACK,
        "total_posts": len(all_posts),
        "total_comments": sum(len(p["comments"]) for p in all_posts),
        "subreddits_scraped": len(SUBREDDITS) - len(failed_subreddits),
        "subreddits_failed": failed_subreddits,
        "posts": all_posts,
    }

    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info("-" * 60)
    log.info(f"✅ Saved {len(all_posts)} posts ({output['total_comments']} comments) → {RAW_DATA_PATH}")
    if failed_subreddits:
        log.warning(f"⚠️  Failed subreddits: {failed_subreddits}")


if __name__ == "__main__":
    main()
