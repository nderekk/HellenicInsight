"""Raw HTML -> article fields (title, text, date, section, is_opinion, fingerprints).

Called by scraper.scrape_site right after a download, and by `reextract` on saved HTML. It only depends on
(html, url, site config), so any fix here or in sites.toml can be applied to old articles without
downloading them again.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import trafilatura  # finds the main text of a web page and drops menus, ads, related links

from .discover import to_utc_iso

if TYPE_CHECKING:                       # only for type hints; a real import would be circular (scraper imports us)
    from .scraper import Site

# Bump when extraction logic changes; `reextract --only-older` can then redo just the rows made by older versions.
EXTRACTOR_VERSION = 1

# URL folders that mean "opinion piece" on Greek sites (efsyn "stiles" = columns, "apopseis" = opinions, ...).
_OPINION_URL = re.compile(r"/(apopseis|apopsi|gnomes|gnomi|opinions?|editorial|arthrografia|stiles|columns?|blogs?)/", re.I)

# Where pages state their publish time, most reliable first:
# <meta property="article:published_time" content="...">, the same with attributes swapped, and JSON-LD.
_META_PUBLISHED = [
    re.compile(r'<meta[^>]+(?:property|name|itemprop)=["\'](?:article:published_time|datePublished|pubdate)["\'][^>]*?content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*?(?:property|name|itemprop)=["\'](?:article:published_time|datePublished)["\']', re.I),
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"'),
]


def extract(html: bytes, url: str, site: Site, listed_published: str | None = None) -> dict:
    """Turn one downloaded page into the article's fields (one `articles` row minus url/source/fetched_at).

    listed_published: the date from the feed/sitemap, used if the page itself doesn't state one.
    """
    doc = trafilatura.bare_extraction(
        html, url=url, with_metadata=True,     # metadata = title, author, date, site name, categories
        favor_precision=True,                  # prefer dropping a paragraph over keeping menus/related links
        include_comments=False, include_tables=False, include_images=False,
        prune_xpath=site.prune_xpath or None,  # per-site page elements to remove first (sites.toml)
    )
    doc = doc.as_dict() if doc else {}         # None when trafilatura finds no article text

    text = doc.get("text") or ""
    for pattern in site.strip:                 # per-site leftovers to remove, e.g. protothema's "Κλείσιμο"
        text = re.sub(pattern, "", text)
    text = _tidy_whitespace(text)
    n_words = len(text.split())

    # Publish date: the page's own meta tag, else the feed's date, else trafilatura's guess.
    head = html[:300_000].decode("utf-8", "ignore")      # meta tags are near the top; no need to scan everything
    published = _meta_published(head) or listed_published or to_utc_iso(doc.get("date"))

    return {
        "title": _clean_title(doc.get("title"), site.name, doc.get("sitename")),
        "author": doc.get("author"),
        "published_at": published,
        "section": _section(url, site.section_regex, doc.get("categories")),
        "is_opinion": int(_is_opinion(url, site)),
        "text": text,
        "n_words": n_words,
        "ok": int(n_words >= site.min_words),          # usable: long enough to be a real article
        "text_sha1": hashlib.sha1(_normalize(text).encode()).hexdigest() if text else None,   # exact duplicates
        "simhash": simhash(text),                          # near duplicates (store.py compares these)
        "extractor_version": EXTRACTOR_VERSION,
    }


def _tidy_whitespace(text: str) -> str:
    """Collapse runs of spaces inside each line and drop empty lines (left behind by `strip` patterns)."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _meta_published(head: str) -> str | None:
    """Publish time from the page's meta tags / JSON-LD, as ISO UTC. None if absent or unparseable."""
    for pattern in _META_PUBLISHED:
        m = pattern.search(head)
        if m:
            iso = to_utc_iso(m.group(1))
            if iso:
                return iso
    return None


def _section(url: str, section_regex: str | None, categories) -> str | None:
    """The site section, e.g. "politiki". From the site's section_regex, else the URL's first folder,
    else (slug-only URLs like thepressproject's) the page's own category."""
    path = urlsplit(url).path
    if section_regex:
        m = re.search(section_regex, path)
        return m.group(1) if m else None
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:                      # /politiki/1496661/slug -> politiki
        return parts[0]
    return categories[0] if categories else None


def _is_opinion(url: str, site: Site) -> bool:
    """Opinion piece if the URL is in a known opinion folder, or matches the site's own opinion_regex."""
    if _OPINION_URL.search(url):
        return True
    return bool(site.opinion_regex and re.search(site.opinion_regex, url))


def _clean_title(title: str | None, site_name: str, sitename: str | None) -> str | None:
    """Remove the site-name suffix: 'Headline - iefimerida.gr' / 'Headline - The Press Project - ...' -> 'Headline'.

    Splits on " - ", " | ", " – " and cuts at the first part that contains the outlet's name
    (from the domain, e.g. "iefimerida", or the page's own site name, e.g. "ΤΑ ΝΕΑ").
    """
    if not title:
        return None
    brands = {_squash(site_name.split(".")[0]), _squash(sitename or "")} - {""}
    parts = re.split(r"\s+[-|–]\s+", title.strip())
    for i in range(1, len(parts)):
        if any(brand in _squash(parts[i]) for brand in brands):
            return " - ".join(parts[:i])
    return title.strip()


def _squash(text: str) -> str:
    """Normalized and without spaces/punctuation, for name matching: "The Press Project" -> "thepressproject"."""
    return re.sub(r"\W", "", _normalize(text))


def _normalize(text: str) -> str:
    """Lowercase, no accents: 'Κυβέρνηση' and 'ΚΥΒΕΡΝΗΣΗ' become the same, so they hash the same."""
    text = unicodedata.normalize("NFD", text.lower())          # split letters from their accents...
    return "".join(c for c in text if unicodedata.category(c) != "Mn")   # ...then drop the accents


def simhash(text: str) -> int:
    """64-bit SimHash over word 3-grams: near-identical texts get hashes that differ in only a few bits.

    How: every 3-word sequence gets a 64-bit hash; each bit position votes +1 if that bit is 1, -1 if 0;
    the final hash has a 1 wherever the votes are positive. Changing a few words only shifts a few votes,
    so one wire story republished on 5 sites (with small edits) ends up with almost the same hash.
    """
    tokens = re.findall(r"\w+", _normalize(text))
    votes = [0] * 64
    for i in range(max(len(tokens) - 2, 0)):
        shingle = " ".join(tokens[i:i + 3]).encode()
        h = int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big")
        for bit in range(64):
            votes[bit] += 1 if h >> bit & 1 else -1
    return sum(1 << bit for bit in range(64) if votes[bit] > 0)
