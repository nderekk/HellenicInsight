"""Feed / sitemap parsing: listing bytes -> article URLs with publish dates.

Takes the bytes of a feed or sitemap (already downloaded by net.py) and returns which articles exist and when
they were published. No network, no database: pure parsing, so it is easy to test on saved files.

The format (RSS, Atom, sitemap, sitemap index) is detected from the content, so sites.toml only lists URLs and
never has to say which kind each one is.

Used by scraper.py (parse_listing, normalize_url, last_number) and extract.py (to_utc_iso).
"""
import re  # regex
import xml.etree.ElementTree as ET  # built-in XML parser, enough for sitemaps
from calendar import timegm  # convert a UTC time tuple into a Unix timestamp
from dataclasses import dataclass  # shortcut for small classes: writes __init__, __repr__, etc.
from datetime import datetime, timezone  # dates/times with timezones
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit  # take URLs apart and put them back together

import feedparser  # external library for RSS/Atom; real feeds are often slightly broken and it copes

# Query parameters that only track where a click came from and never change the page:
_TRACKING_PARAM = re.compile(r"^(utm_\w+|fbclid|gclid|ref)$")


@dataclass
class Listing:
    """What one feed or sitemap file contains.

    A feed or regular sitemap fills `entries`; a sitemap index (a list of other sitemap files) fills `children`.
    """
    # Articles found: (url, published_at). The date is ISO-UTC text, or None when the source gives none.
    # e.g. ("https://www.efsyn.gr/ellada/1496733/...", "2026-10-26T10:07:33+00:00")
    entries: list[tuple[str, str | None]]
    # Sitemap index only: URLs of the sub-sitemaps, newest first. The caller decides how many to download.
    children: list[str]


def normalize_url(url: str) -> str:
    """Give each article exactly one URL, so it is downloaded and stored once.

    The URL is the database key, and "do we have it already?" is a URL lookup. The same article can arrive as
    ".../slug/?utm_source=rss#comments" and as ".../slug/"; both become ".../slug/":
      - tracking parameters are removed, real ones (e.g. "?page=68") are kept;
      - scheme and domain are lowercased (domains are case-insensitive; the path is left alone, it may not be);
      - the "#fragment" is dropped: it only scrolls the page and never reaches the server.
    """
    p = urlsplit(url.strip())  # -> scheme, netloc (domain), path, query, fragment
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING_PARAM.match(k)]
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, urlencode(query), ""))


def to_utc_iso(value) -> str | None:
    """Turn any date we meet into one standard text form, always UTC: '2026-10-26T10:21:37+00:00'.

    Dates arrive as feedparser time tuples, as text from sitemaps/HTML ("2026-10-26T12:21:37+02:00"), or as
    datetime objects. One fixed format, always UTC and whole seconds, means plain text comparison sorts
    chronologically, which scraper.py and health.py rely on (e.g. `published_at >= since`). Returns None for
    missing or unparseable dates: a missing date is acceptable, a crashed run is not.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif hasattr(value, "tm_year"):
        # feedparser's *_parsed fields: a time tuple already converted to UTC. timegm (not time.mktime, which
        # would read it as this computer's local time and shift it by 2-3 hours) turns it into a timestamp.
        dt = datetime.fromtimestamp(timegm(value), tz=timezone.utc)
    else:
        try:
            dt = datetime.fromisoformat(str(value).strip())  # "2026-10-26T12:21:37+02:00", "...Z", "2026-10-26"
        except ValueError:
            return None
    if dt.tzinfo is None:
        # A date without a timezone has to be assumed. Known weak spot: a site writing naive *Greek* local time
        # would end up 2-3 hours off. Our current sources all include the offset.
        dt = dt.replace(tzinfo=timezone.utc)
    # Same moment, written in UTC: 12:21:37+02:00 -> 10:21:37+00:00. No microseconds, so every value has one shape.
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_listing(content: bytes) -> Listing:
    """Read a downloaded feed or sitemap and return its articles (or, for a sitemap index, its sub-sitemaps).

    Takes bytes, not text: the file declares its own encoding (<?xml ... encoding="utf-8"?>) and both parsers
    use that, which keeps Greek correct. Formats are tried in order: RSS/Atom, sitemap index, regular sitemap.
    Anything else (an HTML error page, a Cloudflare challenge) gives an empty Listing: the run then reports
    "listed 0" and the health check turns that into a loud FAIL, instead of the run crashing.
    """
    # 1. RSS/Atom feed. Tried first because feedparser is forgiving: for a sitemap it just finds no entries.
    feed = feedparser.parse(content)
    if feed.entries:
        # Per entry: the link, and the publish date (Atom feeds sometimes only have an update date).
        # Entries without a link are useless and skipped. Feeds never point to other feeds: no children.
        return Listing([(e.link, to_utc_iso(e.get("published_parsed") or e.get("updated_parsed")))
                        for e in feed.entries if e.get("link")], [])

    # 2. Not a feed: try plain XML. `root` is the outermost element, <urlset> or <sitemapindex>.
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return Listing([], [])

    def local(tag):
        # Sitemaps use XML namespaces, so the parser reports tags as
        # "{http://www.sitemaps.org/schemas/sitemap/0.9}loc". Keep only the part after "}" -> "loc".
        return tag.rsplit("}", 1)[-1]

    def fields(node):
        # One sitemap entry -> {"loc": ..., "lastmod": ..., "publication_date": ...}.
        # node.iter() also walks nested elements, needed for Google News sitemaps, where the date sits two
        # levels deep (<news:news><news:publication_date>).
        # setdefault = first occurrence wins. Required: an entry can contain an image block with its own
        # <image:loc>, which comes after the article's <loc>. Overwriting made backfill see .jpg files as articles.
        out = {}
        for child in node.iter():
            out.setdefault(local(child.tag), (child.text or "").strip())
        return out

    # 3. Sitemap index: a list of sitemap files, not articles.
    if local(root.tag) == "sitemapindex":
        children = [fields(s) for s in root if local(s.tag) == "sitemap"]
        # Newest first, because backfill walks back in time and stops at --days. Sites differ, so the sort key
        # is a pair: lastmod first (efsyn dates every file), then the last number in the URL for ties or missing
        # dates (protothema has no dates but numbers its files .../78.xml; iefimerida gives every file the same
        # date but ?page=68 is newest). ISO dates sort correctly as text.
        children.sort(key=lambda c: (c.get("lastmod", ""), last_number(c.get("loc", ""))), reverse=True)
        return Listing([], [c["loc"] for c in children if c.get("loc")])

    # 4. Regular sitemap: one <url> per article. Prefer publication_date (Google News sitemaps, the real
    #    publish time) over lastmod (last *modification*, later than publication if the article was edited).
    #    extract.py prefers the date printed on the article page over both.
    #    Valid XML that is neither kind has no <url> elements, so it also ends up as an empty Listing.
    entries = []
    for node in root:
        if local(node.tag) == "url":
            f = fields(node)
            if f.get("loc"):
                entries.append((f["loc"], to_utc_iso(f.get("publication_date") or f.get("lastmod"))))
    return Listing(entries, [])


def last_number(url: str) -> int:
    """The last run of digits in a URL, as an int (-1 if none, so those sort last).

    ".../NewsArticles/78.xml" -> 78, "...sitemap.xml?page=68" -> 68, ".../politiki/1496661/slug/" -> 1496661.
    Converted to int because as text "9" > "78". Used to order sub-sitemaps (above) and undated article URLs
    (scraper.newest_first: protothema lists articles oldest-first without dates, and higher IDs are newer).
    """
    numbers = re.findall(r"\d+", url)
    return int(numbers[-1]) if numbers else -1
