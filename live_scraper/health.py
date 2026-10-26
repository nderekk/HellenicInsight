"""Health checks. Each returns Findings; any FAIL makes `run` exit non-zero so cron/systemd notice.

Called by scraper.report_health after every run, `reextract` and `health`. All checks read the database.

  volume   the feed listed nothing, every extraction failed, or no new good article for `stale_hours`
  length   median words of the last 24h fell below half of the previous two weeks' median
  phrases  a phrase is in >= 20% of one outlet's recent articles but <= 2% of the other outlets'
           (leftover page text, like protothema's "Κλείσιμο" button, or a byline/signature)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING

from .store import Store, days_ago_iso

if TYPE_CHECKING:                       # only for type hints; a real import would be circular (scraper imports us)
    from .scraper import Site

PHRASE_SHARE = 0.20          # flag phrases in >= 20% of an outlet's articles...
PHRASE_OTHER_MAX = 0.02      # ...that appear in <= 2% of the other outlets' articles
PHRASE_SAMPLE = 300          # look at each outlet's 300 most recent articles
PHRASE_MIN_ARTICLES = 30     # too few articles = percentages mean nothing, skip
LENGTH_MIN_ARTICLES = 10
LENGTH_DROP_RATIO = 0.5


@dataclass
class Finding:
    """One check's result for one outlet. Saved in the `health` table; FAILs are also logged."""
    source: str
    check: str          # "volume", "length" or "phrases"
    level: str          # "FAIL" or "OK"
    message: str


def check_all(store: Store, sites: list[Site], run_stats: dict | None = None) -> list[Finding]:
    """Run every check for every site. run_stats = this run's counts per site (None when no scrape ran)."""
    run_stats = run_stats or {}
    findings = []
    for site in sites:
        findings += check_volume(store, site, run_stats.get(site.name))
        findings += check_length(store, site)
    findings += check_phrases(store, sites)       # compares outlets with each other, so all at once
    return findings


def check_volume(store: Store, site: Site, stats: dict | None) -> list[Finding]:
    """Is the outlet still giving us articles?

    From this run's counts (if there was a run): an error, an empty listing, or downloads with no usable text.
    From the database (always): no new usable article for `stale_hours`.
    """
    out = []
    if stats is not None:
        if stats.get("error"):
            out.append(Finding(site.name, "volume", "FAIL", f"run error: {stats['error']}"))
        elif stats.get("listed", 0) == 0:
            out.append(Finding(site.name, "volume", "FAIL", "listing returned 0 articles (feed/sitemap URL moved or blocked?)"))
        elif stats.get("fetched", 0) > 0 and stats.get("ok", 0) == 0:
            out.append(Finding(site.name, "volume", "FAIL",
                               f"0 of {stats['fetched']} downloads gave usable text (layout change?)"))

    last = store.query("SELECT MAX(fetched_at) FROM articles WHERE source = ? AND ok = 1", (site.name,))[0][0]
    if last is None or last < days_ago_iso(site.stale_hours / 24):
        out.append(Finding(site.name, "volume", "FAIL", f"no new usable article since {last or 'ever'}"))
    return out or [Finding(site.name, "volume", "OK", "")]


def check_length(store: Store, site: Site) -> list[Finding]:
    """Did articles suddenly get much shorter? Compares the last 24h's median length with the 14 days before.

    Uses all rows, not just ok ones: a collapse to near-empty text is exactly what this should catch.
    """
    rows = store.query("SELECT n_words, fetched_at >= ? AS recent FROM articles WHERE source = ? AND fetched_at >= ?",
                       (days_ago_iso(1), site.name, days_ago_iso(15)))
    recent = [r["n_words"] for r in rows if r["recent"]]
    before = [r["n_words"] for r in rows if not r["recent"]]
    if len(recent) < LENGTH_MIN_ARTICLES or len(before) < LENGTH_MIN_ARTICLES:
        return [Finding(site.name, "length", "OK", "not enough articles to compare yet")]

    now, then = median(recent), median(before)
    if now < LENGTH_DROP_RATIO * then:
        return [Finding(site.name, "length", "FAIL", f"median words fell from {then:.0f} to {now:.0f}")]
    return [Finding(site.name, "length", "OK", f"median words {now:.0f} (before {then:.0f})")]


def _phrases(text: str, max_words: int = 4) -> set[str]:
    """Every 1- to 4-word sequence in the text, as a set (we count articles containing it, not occurrences).

    Case kept on purpose: the button "Κλείσιμο" must not be confused with the word "κλείσιμο" (closure).
    """
    tokens = re.findall(r"\w+", text)
    phrases = set()
    for n in range(1, max_words + 1):
        for i in range(len(tokens) - n + 1):
            phrases.add(" ".join(tokens[i:i + n]))
    return phrases


def check_phrases(store: Store, sites: list[Site]) -> list[Finding]:
    """Find text that only one outlet has: leftover page junk, or anything that lets a model spot the outlet.

    Per outlet: phrases in >= 20% of its recent articles (candidates), kept only if <= 2% of the other
    outlets' articles have them (common Greek phrases appear everywhere, so they drop out). Phrases in the
    site's allow_phrases (and longer phrases containing them) are accepted.
    """
    texts = {}
    for s in sites:
        rows = store.query("SELECT text FROM articles WHERE source = ? AND ok = 1 ORDER BY fetched_at DESC LIMIT ?",
                           (s.name, PHRASE_SAMPLE))
        texts[s.name] = [r["text"] for r in rows]

    out = []
    for site in sites:
        mine = texts[site.name]
        others = [t for name, outlet_texts in texts.items() if name != site.name for t in outlet_texts]
        if len(mine) < PHRASE_MIN_ARTICLES or len(others) < PHRASE_MIN_ARTICLES:
            continue

        # 1. Candidates: phrases in >= 20% of this outlet's articles, minus allowed ones.
        articles_with = Counter(p for t in mine for p in _phrases(t))
        candidates = {}
        for phrase, count in articles_with.items():
            share = count / len(mine)
            allowed = any(a in phrase for a in site.allow_phrases)
            if share >= PHRASE_SHARE and not allowed:
                candidates[phrase] = share
        if not candidates:
            out.append(Finding(site.name, "phrases", "OK", ""))
            continue

        # 2. Keep candidates that are rare at the other outlets.
        others_with = Counter()
        for t in others:
            others_with.update(candidates.keys() & _phrases(t))
        flagged = {p: share for p, share in candidates.items() if others_with[p] / len(others) <= PHRASE_OTHER_MAX}

        # 3. Report the longest phrases first and skip the fragments they contain.
        shown = []
        for p in sorted(flagged, key=lambda p: (-len(p.split()), -flagged[p])):
            if not any(p in q for q in shown):
                shown.append(p)
        if shown:
            listing = "; ".join(f'"{p}" {flagged[p]:.0%}' for p in shown[:5])
            out.append(Finding(site.name, "phrases", "FAIL",
                               f"outlet-only phrases: {listing}. Strip them in sites.toml (or allow_phrases), then `reextract`"))
        else:
            out.append(Finding(site.name, "phrases", "OK", ""))
    return out
