# briefing.py
# NEW FILE — core of the refactored QuickPulse.
#
# For each topic bucket this module makes ONE Groq API call that returns:
#   - a 3-sentence briefing synthesising all headlines in that bucket
#   - a topic-level sentiment label (Mostly Positive / Mixed / Mostly Negative)
#   - up to 5 named entities (companies, people, countries)
#
# The full briefing result is cached to cache.json for 60 minutes so that
# subsequent page loads are instant.

import json
import os
import time
from pathlib import Path

from groq import Groq

from cluster_news import TOPIC_LABELS, extract_keywords

# ── Configuration ─────────────────────────────────────────────────────────────

CACHE_FILE = Path("cache.json")
CACHE_TTL_SECONDS = 60 * 60          # 60 minutes
GROQ_MODEL = "llama-3.1-8b-instant"  # fast, free-tier friendly
MAX_ARTICLES_PER_TOPIC = 12          # cap to keep prompt short

TOPIC_EMOJIS = {
    "Artificial Intelligence": "🤖",
    "Business & Finance": "📈",
    "Politics & World Affairs": "🌍",
    "Science & Technology": "🔬",
    "Health & Medicine": "🏥",
    "Environment & Climate": "🌱",
}

# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache() -> dict | None:
    """Return cached briefing if it exists and is fresh, else None."""
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        age = time.time() - data.get("timestamp", 0)
        if age < CACHE_TTL_SECONDS:
            print(f"Cache hit — {int(age)}s old (TTL {CACHE_TTL_SECONDS}s).")
            return data
        print(f"Cache stale ({int(age)}s old). Regenerating.")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _save_cache(briefing: dict) -> None:
    briefing["timestamp"] = time.time()
    CACHE_FILE.write_text(json.dumps(briefing, indent=2))
    print("Briefing cached.")


# ── Groq synthesis ─────────────────────────────────────────────────────────────

def _build_prompt(topic: str, articles: list[dict]) -> str:
    headlines_block = "\n".join(
        f"- {a['title']}. {a.get('snippet', '')}".strip()
        for a in articles[:MAX_ARTICLES_PER_TOPIC]
    )
    return f"""You are a senior news editor writing a daily intelligence briefing.

Topic: {topic}
Headlines ({len(articles[:MAX_ARTICLES_PER_TOPIC])} articles):
{headlines_block}

Return ONLY a JSON object with these exact keys:
{{
  "briefing": "<3 sentences synthesising the key developments. Do NOT copy any headline verbatim. Write in your own words.>",
  "sentiment": "<one of: Mostly Positive | Mixed | Mostly Negative>",
  "entities": ["<entity1>", "<entity2>", "<entity3>", "<entity4>", "<entity5>"],
  "top_story": "<title of the single most significant article from the list above>"
}}

Rules:
- briefing must be exactly 3 sentences.
- entities are the most-mentioned organisations, people, or countries.
- Return ONLY the JSON. No markdown fences, no preamble."""


def _call_groq(topic: str, articles: list[dict]) -> dict:
    """Call Groq and parse the JSON response for one topic."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    prompt = _build_prompt(topic, articles)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        # Strip accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"JSON parse error for topic '{topic}': {e}\nRaw: {raw[:200]}")
        return _fallback(topic)
    except Exception as e:
        print(f"Groq call failed for topic '{topic}': {e}")
        return _fallback(topic)


def _fallback(topic: str) -> dict:
    return {
        "briefing": "Briefing unavailable for this topic.",
        "sentiment": "Mixed",
        "entities": [],
        "top_story": "",
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_briefing(
    topic_buckets: dict[str, list[dict]],
    force: bool = False,
    divergence_result: dict | None = None,   # add this
) -> dict:
    ...
    # Before _save_cache(result):
    if divergence_result:
        result["divergence"] = divergence_result
    _save_cache(result)
    return result


def get_last_updated(briefing: dict) -> str:
    """Return a human-readable 'last updated' string."""
    ts = briefing.get("timestamp")
    if not ts:
        return "Unknown"
    age_seconds = int(time.time() - ts)
    if age_seconds < 60:
        return "Just now"
    if age_seconds < 3600:
        return f"{age_seconds // 60} min ago"
    return f"{age_seconds // 3600}h ago"
