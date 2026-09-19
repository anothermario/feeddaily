"""
FeedDaily — Daily news & podcast briefing generator.
Fetches RSS feeds and YouTube transcripts, then summarizes via Claude.
"""

import os
import logging
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import Optional

import feedparser
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from anthropic import Anthropic

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TRANSCRIPT_MAX_CHARS = 8000

RSS_FEEDS = {
    # Geopolitics & World News
    "BBC World":          "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera":         "https://www.aljazeera.com/xml/rss/all.xml",
    "Foreign Policy":     "https://foreignpolicy.com/feed/",
    "DW News":            "https://rss.dw.com/xml/rss-en-world",
    "The Guardian World": "https://www.theguardian.com/world/rss",

    # Austrian News
    "ORF News":           "https://rss.orf.at/news.xml",
    "Der Standard":       "https://www.derstandard.at/rss",
    "Die Presse":         "https://www.diepresse.com/rss",
    "VOL.AT":             "https://www.vol.at/rss",
    "The Local Austria":  "https://www.thelocal.at/feed/",

    # Finance
    "Yahoo Finance":      "https://finance.yahoo.com/news/rssindex",
    "BBC Business":       "https://feeds.bbci.co.uk/news/business/rss.xml",
    "CNBC":               "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch":        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "The Economist":      "https://www.economist.com/finance-and-economics/rss.xml",

    # AI & Tech
    "The Verge AI":       "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
    "Ars Technica":       "https://feeds.arstechnica.com/arstechnica/index",
    "TechCrunch AI":      "https://techcrunch.com/category/artificial-intelligence/feed/",
    "MIT Tech Review":    "https://www.technologyreview.com/feed/",
    "TLDR AI":            "https://tldr.tech/api/rss/ai",

    # Tennis
    "BBC Tennis":         "https://feeds.bbci.co.uk/sport/tennis/rss.xml",
    "Guardian Tennis":    "https://www.theguardian.com/sport/tennis/rss",
    "Eurosport Tennis":   "https://www.eurosport.com/tennis/rss.xml",
    "Tennis World USA":   "https://www.tennisworldusa.org/rss/all.rss",
    "Tennis.com":         "https://www.tennis.com/news/feed/rss/",

    # Podcasts (episode descriptions)
    "Lex Fridman":        "https://lexfridman.com/feed/podcast/",
    "Joe Rogan (JRE)":    "https://feeds.megaphone.fm/GLT1412515089",
}

YOUTUBE_IDS: list[str] = [
    # Add specific video IDs here when you want a video summarized
    # Example: "dQw4w9WgXcQ"
]

ARTICLES_PER_FEED = 3  # How many articles to pull per feed


# ── RSS Ingestion ─────────────────────────────────────────────────────────────
def fetch_rss_feeds(feeds: dict[str, str]) -> list[dict]:
    """
    Fetch top articles from each RSS feed.

    Args:
        feeds: Dict of {feed_name: feed_url}

    Returns:
        List of article dicts with keys: source, title, link, summary
    """
    articles = []
    for name, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries[:ARTICLES_PER_FEED]:
                articles.append({
                    "source":  name,
                    "title":   entry.get("title", "No title"),
                    "link":    entry.get("link", ""),
                    "summary": entry.get("summary", "") or entry.get("description", ""),
                })
                count += 1
            log.info(f"  ✓ {name} — {count} articles")
        except Exception as e:
            log.warning(f"  ✗ {name} — failed: {e}")
    return articles


# ── YouTube Transcripts ───────────────────────────────────────────────────────
def fetch_transcripts(video_ids: list[str]) -> list[dict]:
    """
    Fetch and truncate YouTube transcripts.

    Args:
        video_ids: List of YouTube video IDs

    Returns:
        List of dicts with keys: video_id, transcript
    """
    results = []
    for vid in video_ids:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(vid)
            text = " ".join(item["text"] for item in transcript)
            if len(text) > TRANSCRIPT_MAX_CHARS:
                text = text[:TRANSCRIPT_MAX_CHARS] + "... [truncated]"
            results.append({"video_id": vid, "transcript": text})
            log.info(f"  ✓ YouTube {vid} — {len(text)} chars")
        except (TranscriptsDisabled, NoTranscriptFound):
            log.warning(f"  ✗ YouTube {vid} — no transcript available")
        except Exception as e:
            log.warning(f"  ✗ YouTube {vid} — error: {e}")
    return results


# ── Context Builder ───────────────────────────────────────────────────────────
def build_context(articles: list[dict], transcripts: list[dict]) -> str:
    """
    Format all collected data into a single context string for the LLM.

    Args:
        articles:    List of RSS article dicts
        transcripts: List of YouTube transcript dicts

    Returns:
        Formatted context string
    """
    lines = ["## RSS Articles & Podcast Episodes\n"]
    for a in articles:
        lines.append(f"**Source:** {a['source']}")
        lines.append(f"**Title:** {a['title']}")
        lines.append(f"**Summary:** {a['summary'][:500]}\n")

    if transcripts:
        lines.append("\n## YouTube Transcripts\n")
        for t in transcripts:
            lines.append(f"**Video ID:** {t['video_id']}")
            lines.append(f"**Transcript:** {t['transcript']}\n")

    return "\n".join(lines)


# ── LLM Summarization ─────────────────────────────────────────────────────────
def generate_briefing(context: str, api_key: str) -> str:
    """
    Send context to Claude and return a structured Markdown briefing.

    Args:
        context: Formatted news + transcript context
        api_key: Anthropic API key

    Returns:
        Markdown briefing string
    """
    client = Anthropic(api_key=api_key)
    today = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""You are a personal news analyst. Your job is to create a focused daily briefing.

RULES:
- Be concise. No filler. No repetition.
- Strip all sponsorships, ads, and self-promotion from podcast/video content.
- Focus on what actually matters and is actionable or important to know.
- Group Austrian news separately since it is local context.
- For tennis: only include results, rankings, or significant news — skip fluff.

OUTPUT FORMAT (strict Markdown):

# 📰 Daily Briefing — {today}

## 🔑 Top 3 Takeaways
- [Most important thing happening globally today — 1-2 sentences]
- [Second most important — 1-2 sentences]
- [Third most important — 1-2 sentences]

## 🌍 World & Geopolitics
| Source | Headline | What it means |
|--------|----------|---------------|
| ... | ... | ... |

## 🇦🇹 Austria
| Source | Headline | What it means |
|--------|----------|---------------|
| ... | ... | ... |

## 💰 Finance & Economy
| Source | Headline | What it means |
|--------|----------|---------------|
| ... | ... | ... |

## 🤖 AI & Tech
| Source | Headline | What it means |
|--------|----------|---------------|
| ... | ... | ... |

## 🎾 Tennis
| Source | Headline | What it means |
|--------|----------|---------------|
| ... | ... | ... |

## 🎙️ Podcasts & Videos
[For each source with content: bullet points of key insights only]

---
*Generated by FeedDaily on {today}*

---

HERE IS TODAY'S DATA:

{context}
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("  FeedDaily — starting daily briefing")
    log.info("=" * 55)

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Run: $env:ANTHROPIC_API_KEY='sk-ant-...'")

    # Step 1 — Fetch RSS
    log.info("\n[1/4] Fetching RSS feeds...")
    articles = fetch_rss_feeds(RSS_FEEDS)
    log.info(f"      Total: {len(articles)} articles collected")

    # Step 2 — Fetch YouTube transcripts
    log.info("\n[2/4] Fetching YouTube transcripts...")
    transcripts = fetch_transcripts(YOUTUBE_IDS)
    log.info(f"      Total: {len(transcripts)} transcripts collected")

    # Step 3 — Build context
    log.info("\n[3/4] Building context...")
    context = build_context(articles, transcripts)
    log.info(f"      Context size: {len(context)} characters")

    # Step 4 — Generate briefing
    log.info("\n[4/4] Generating briefing with Claude...")
    briefing = generate_briefing(context, api_key)

        # Output — save HTML and send via email
    import markdown
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    html_content = markdown.markdown(briefing, extensions=["tables"])

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FeedDaily — {datetime.now().strftime("%Y-%m-%d")}</title>
<style>
  body {{
    background: #0f1117;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px;
    line-height: 1.7;
    padding: 2rem 1rem;
    max-width: 900px;
    margin: 0 auto;
  }}
  h1 {{ font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.15rem; font-weight: 600; color: #6366f1; margin: 2rem 0 0.75rem;
        padding-bottom: 0.4rem; border-bottom: 1px solid #2e3147; }}
  h3 {{ font-size: 1rem; font-weight: 600; color: white; margin: 1.25rem 0 0.5rem; }}
  p {{ margin-bottom: 0.75rem; }}
  ul {{ padding-left: 1.4rem; margin-bottom: 0.75rem; }}
  li {{ margin-bottom: 0.4rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.75rem 0 1.5rem; font-size: 0.9rem; }}
  th {{ background: #2e3147; color: #8892a4; text-align: left; padding: 0.5rem 0.75rem;
        font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }}
  td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid #2e3147; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  strong {{ color: white; }}
  hr {{ border: none; border-top: 1px solid #2e3147; margin: 2rem 0; }}
</style>
</head>
<body>
{html_content}
<p style="color:#8892a4;font-size:0.8rem;text-align:center;margin-top:3rem;">
Generated by FeedDaily · {datetime.now().strftime("%Y-%m-%d %H:%M")}
</p>
</body>
</html>"""

    # Save briefing to docs folder
    import pathlib
    import json

    pathlib.Path("docs").mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    output_file = f"docs/briefing_{today}.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_page)
    log.info(f"      Saved: {output_file}")

    # Update archive index
    archive_file = pathlib.Path("docs/archive.json")
    if archive_file.exists():
        dates = json.loads(archive_file.read_text())
    else:
        dates = []
    if today not in dates:
        dates.append(today)
        dates.sort()
    archive_file.write_text(json.dumps(dates))
    log.info(f"      Archive updated: {len(dates)} briefings")
    log.info(f"\n✅ Done!")
    log.info("=" * 55)