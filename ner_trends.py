# ner_trends.py  

from collections import Counter, defaultdict
from typing import Optional
import subprocess, sys
try:
    import spacy
    spacy.load("en_core_web_sm")
except (ImportError, OSError):
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                   check=True)

# en_core_web_sm is ~12MB — fast on CPU, no GPU needed
# Install: python -m spacy download en_core_web_sm
_nlp: Optional[spacy.language.Language] = None

# Entity types we care about
_KEEP_LABELS = {"ORG", "PERSON", "GPE", "LOC"}

# Common noise entities to suppress (appear in every news cycle, add no signal)
_STOPLIST = {
    "reuters", "ap", "bloomberg", "associated press", "the new york times",
    "bbc", "cnn", "fox news", "the guardian", "wsj", "washington post",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "u.s.", "us", "united states",  # keep "US" for GPE but suppress generic
}

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp


def extract_trending_entities(
    articles: list[dict],
    top_n: int = 8,
) -> dict[str, list[dict]]:
    """
    Run NER over all article titles+snippets and return frequency-ranked
    entities split by type.

    Returns:
    {
        "ORG":    [{"text": "OpenAI", "count": 8, "articles": [...]}, ...],
        "PERSON": [{"text": "Sam Altman", "count": 4, "articles": [...]}, ...],
        "GPE":    [{"text": "China", "count": 7, "articles": [...]}, ...],
    }
    """
    nlp = _get_nlp()

    # Build texts and keep index → article mapping
    texts = [
        f"{a.get('title', '')}. {a.get('snippet', '')}".strip()
        for a in articles
    ]

    # entity_text_lower → {label, canonical_text, set of article indices}
    entity_map: dict[str, dict] = {}

    # Batch process — much faster than one doc at a time
    for idx, doc in enumerate(nlp.pipe(texts, batch_size=32)):
        for ent in doc.ents:
            if ent.label_ not in _KEEP_LABELS:
                continue
            normalized = ent.text.strip()
            key = normalized.lower()
            if key in _STOPLIST or len(key) < 2:
                continue
            if key not in entity_map:
                entity_map[key] = {
                    "text": normalized,
                    "label": ent.label_,
                    "article_indices": set(),
                }
            # Use the longest seen form as canonical
            if len(normalized) > len(entity_map[key]["text"]):
                entity_map[key]["text"] = normalized
            entity_map[key]["article_indices"].add(idx)

    # Group by label, sort by article count descending
    by_label: dict[str, list[dict]] = defaultdict(list)
    for key, info in entity_map.items():
        count = len(info["article_indices"])
        # Slim article list: just title + url for the UI
        article_refs = [
            {"title": articles[i]["title"], "url": articles[i].get("url", "#")}
            for i in sorted(info["article_indices"])[:3]  # max 3 refs per entity
        ]
        by_label[info["label"]].append({
            "text": info["text"],
            "count": count,
            "articles": article_refs,
        })

    # Sort each bucket by count, take top N
    result = {}
    for label in ["ORG", "PERSON", "GPE"]:
        bucket = sorted(by_label.get(label, []), key=lambda x: x["count"], reverse=True)
        result[label] = bucket[:top_n]

    return result


def entity_summary_for_briefing(trending: dict[str, list[dict]]) -> str:
    """
    Returns a one-line plain-text summary of top entities.
    Used to inject entity context into the Groq prompt.
    """
    parts = []
    for label, entities in trending.items():
        if entities:
            names = ", ".join(e["text"] for e in entities[:3])
            parts.append(f"{label}: {names}")
    return " | ".join(parts)