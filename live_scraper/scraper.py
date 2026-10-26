"""HellenicInsight news scraper.

Polls Greek news outlets on an interval, stores every article with its raw HTML, and checks its own health.
Adding or fixing an outlet is a sites.toml edit; the code is the same for every site.

Design
  1. Find articles through RSS and sitemaps, not page crawling. Feeds change far less often than page layouts.
  2. Extract text with trafilatura, with a small optional per-site config, instead of hand-written regex.
  3. Save the raw HTML. When extraction is wrong, fix it and re-extract offline (`reextract`) without scraping again.
  4. Automatic checks that fail loudly (health.py):
       - an outlet returns 0 articles
       - median text length drops sharply
       - one phrase appears in more than ~20% of an outlet's articles
  5. Be polite to the sites: respect robots.txt, rate-limit each site, and identify ourselves in the User-Agent.
  6. Store metadata from the start: published_at, section, is_opinion, and a hash for removing near-duplicates.

Flow per site (one thread each):  feeds -> new URLs -> fetch (net.py) -> raw HTML + extract (extract.py) -> SQLite (store.py)

This file is the coordinator: it reads sites.toml, runs the flow above for each outlet, and holds the CLI commands.
The actual work is done by the helper modules (net, discover, extract, store, health).

Usage (from the repo root):
  python -m live_scraper run                     one pass over all enabled sites, then health checks
  python -m live_scraper loop                    catch up on the gap since the last run, then run every
                                                 `loop_minutes` forever (prunes raw HTML daily)
  python -m live_scraper backfill --days 30 --limit 300 --include "/(politiki|politics)/"
  python -m live_scraper reextract --sites protothema.gr      re-run extraction on stored HTML
  python -m live_scraper prune --raw-days 30     delete raw HTML older than 30 days (text stays in the DB)
  python -m live_scraper run --sites efsyn.gr --limit 3 && python -m live_scraper show -n 3   quick check
  python -m live_scraper health | stats
  python -m live_scraper export --hours 24       write kafka_feed.jsonl for backend/producer.py
"""
import argparse  # command-line parsing (run, loop, backfill, ...)
import json
import logging
import re
import sys
import time
import tomllib  # reads sites.toml (built into Python 3.11+)
from concurrent.futures import ThreadPoolExecutor  # one thread per outlet
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from . import health
from .discover import last_number, normalize_url, parse_listing
from .extract import EXTRACTOR_VERSION, extract
from .net import Fetcher
from .store import Store, days_ago_iso, now_iso

log = logging.getLogger("scraper")

PACKAGE_DIR = Path(__file__).resolve().parent  # .../live_scraper
REPO_ROOT = PACKAGE_DIR.parent                 # data_dir in sites.toml is relative to this
BACKFILL_MAX_PAGES = 50                        # safety cap on feed pages / sub-sitemaps per backfill source
CATCHUP_MIN_GAP_HOURS = 1                      # shorter gaps need no catch-up: every feed covers 2+ hours

# How scrape_site finds URLs: "live" reads the feeds, "backfill" walks older pages/sitemaps.
Mode = Literal["live", "backfill"]
MODES = ("live", "backfill")


@dataclass
class Site:
    """One outlet's settings, built from a [[site]] block in sites.toml (merged over [defaults]).

    Passed to every step: discovery (feeds, backfill, url filters), extraction (strip, prune_xpath, regexes),
    and health (allow_phrases, stale_hours). The defaults below match [defaults] in sites.toml, which wins.
    """
    name: str                        # domain, also the `source` column, e.g. "efsyn.gr"
    feeds: list[str]                 # polled on every live run
    backfill: list[str] = field(default_factory=list)   # used by `backfill`; "{page}" = page number
    enabled: bool = True
    url_include: str | None = None   # regex: only matching article URLs are kept
    url_exclude: str | None = None   # regex: matching article URLs are skipped
    section_regex: str | None = None
    opinion_regex: str | None = None
    prune_xpath: list[str] = field(default_factory=list)
    strip: list[str] = field(default_factory=list)
    allow_phrases: list[str] = field(default_factory=list)
    delay_seconds: float = 3.0
    min_words: int = 50
    max_new_per_run: int = 200
    stale_hours: float = 12


@dataclass
class Settings:
    """Global settings ([settings] in sites.toml) plus the list of all outlets, enabled or not."""
    user_agent: str
    data_dir: Path
    loop_minutes: float
    raw_keep_days: float
    catchup_max_days: float      # `loop` startup catch-up: never look back further than this
    catchup_limit: int           # ...nor download more than this many articles per site
    sites: list[Site]


def load_settings(path: Path = PACKAGE_DIR / "sites.toml") -> Settings:
    """Read sites.toml into Settings. Called once at startup by main().

    Each [[site]] block is merged over [defaults], so a site only lists what differs from the defaults.
    """
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    s, defaults = cfg["settings"], cfg.get("defaults", {})
    sites = [Site(**{**defaults, **block}) for block in cfg.get("site", [])]  # block keys override defaults
    return Settings(s["user_agent"], REPO_ROOT / s["data_dir"], s.get("loop_minutes", 15),
                    s.get("raw_keep_days", 30), s.get("catchup_max_days", 3), s.get("catchup_limit", 500), sites)


# ── discovery: which articles exist ──────────────────────────────────────────

def fetch_listing(fetcher: Fetcher, url: str):
    """Download one feed/sitemap and parse it (discover.parse_listing). None if robots.txt forbids it or it fails."""
    r = fetcher.get(url)
    if r is None or r.status_code != 200:
        log.warning("listing %s -> %s", url, "robots.txt" if r is None else r.status_code)
        return None
    return parse_listing(r.content)


def read_listing(fetcher: Fetcher, url: str, depth: int = 0) -> list[tuple[str, str | None]]:
    """Feed or sitemap URL -> [(article url, published_at)]. Used by live runs.

    For a sitemap index it follows only the 2 newest sub-sitemaps (a live run only needs recent articles);
    `depth` stops it from going more than 2 levels deep.
    """
    listing = fetch_listing(fetcher, url)
    if listing is None:
        return []
    entries = list(listing.entries)
    if depth < 2:
        for child in listing.children[:2]:
            entries += read_listing(fetcher, child, depth + 1)
    return entries


def backfill_pages(fetcher: Fetcher, source: str):
    """Used by backfill: yields batches of entries, newest first, so the caller can stop as soon as it has enough.

    A "{page}" source (paged feed, e.g. efsyn's /feed/?paged={page}) is read page by page; a sitemap index
    (e.g. protothema's) sub-sitemap by sub-sitemap. A generator, so nothing is downloaded until it's asked for.
    """
    if "{page}" in source:
        for page in range(1, BACKFILL_MAX_PAGES + 1):
            yield read_listing(fetcher, source.format(page=page))
        return
    listing = fetch_listing(fetcher, source)
    if listing is None:
        return
    if listing.entries:
        yield listing.entries
    for child in listing.children[:BACKFILL_MAX_PAGES]:
        yield read_listing(fetcher, child, depth=1)


def keep(site: Site, url: str, include: str | None) -> bool:
    """Decide whether an article URL is scraped for this site: same domain, not excluded, matches includes.

    `include` is the extra regex from `backfill --include` (e.g. politics only).
    """
    host = urlsplit(url).netloc
    if not (host == site.name or host.endswith("." + site.name)):     # feeds sometimes link to sister sites
        return False
    if site.url_exclude and re.search(site.url_exclude, url):
        return False
    return all(re.search(p, url) for p in (site.url_include, include) if p)


def newest_first(entries: dict[str, str | None]) -> list[str]:
    """Sort article URLs newest first, so per-run limits keep the most recent ones.

    By date; undated sitemap URLs fall back to the article ID in the URL (higher = newer).
    """
    return sorted(entries, key=lambda u: (entries[u] or "", last_number(u)), reverse=True)


def backfill_listing(fetcher: Fetcher, site: Site, since: str, include: str | None, limit: int,
                     store: Store) -> dict[str, str | None]:
    """Collect older article URLs for one site from its `backfill` sources. Used by `backfill` instead of feeds.

    Walks pages/sub-sitemaps newest first and stops when one of these happens:
    a page brings nothing new (end reached, or the site ignores ?page=), everything on a page is older than
    `since`, or there are already `limit` wanted URLs we don't have yet. Returns {url: published_at}.
    """
    found: dict[str, str | None] = {}
    for source in site.backfill:
        for entries in backfill_pages(fetcher, source):
            entries = {normalize_url(u): d for u, d in entries}
            fresh = {u: d for u, d in entries.items() if u not in found}
            if not fresh:                                                # past the end, or the site ignores ?page=
                break
            found.update(fresh)

            dates_on_page = [d for d in fresh.values() if d]
            page_is_too_old = bool(dates_on_page) and max(dates_on_page) < since

            wanted = [u for u in found if keep(site, u, include)]
            missing = set(wanted) - store.known_urls(wanted)
            have_enough = len(missing) >= limit

            if page_is_too_old or have_enough:
                break
    return {u: d for u, d in found.items() if not d or d >= since}   # undated ones are kept: their age is unknown


# ── scraping: the per-site flow ──────────────────────────────────────────────

def scrape_site(site: Site, settings: Settings, store: Store, mode: Mode = "live",
                since: str | None = None, limit: int | None = None, include: str | None = None) -> dict:
    """The whole flow for one outlet. Runs in its own thread (see run_once). Returns counts for the summary.

    1. list article URLs (feeds in "live" mode, backfill sources in "backfill" mode)
    2. keep this site's URLs that aren't in the database yet, newest first, capped per run
    3. for each: download (net) -> extract -> save raw HTML + row (store)
    Errors on one article are counted and skipped; an error in the listing step is recorded in stats["error"],
    which the health check reports. The run itself is logged in the `runs` table either way.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    stats = {"listed": 0, "new": 0, "fetched": 0, "ok": 0, "errors": 0}
    started = now_iso()
    fetcher = Fetcher(settings.user_agent, site.delay_seconds)          # own fetcher = own rate limit per site
    try:
        if mode == "live":
            listed = {normalize_url(u): d for feed in site.feeds for u, d in read_listing(fetcher, feed)}
        else:
            listed = backfill_listing(fetcher, site, since, include, limit, store)
        listed = {u: d for u, d in listed.items() if keep(site, u, include)}
        known = store.known_urls(listed)
        new = [u for u in newest_first(listed) if u not in known][: limit or site.max_new_per_run]
        stats.update(listed=len(listed), new=len(new))
        log.info("%s: %d listed, %d new", site.name, len(listed), len(new))

        for url in new:
            try:
                r = fetcher.get(url)
                if r is None or r.status_code != 200:                    # None = robots.txt forbids it
                    stats["errors"] += 1
                    log.warning("%s -> %s", url, "robots.txt" if r is None else r.status_code)
                    continue
                row = extract(r.content, url, site, listed[url])         # listed[url] = feed date, as fallback
                row.update(url=url, source=site.name, fetched_at=now_iso(), http_status=r.status_code,
                           raw_path=store.save_raw(site.name, url, r.content))
                store.save_article(row)
                stats["fetched"] += 1
                stats["ok"] += row["ok"]
            except Exception as e:                                       # one bad page never stops the run
                stats["errors"] += 1
                log.warning("%s failed: %r", url, e)
    except Exception as e:
        stats["error"] = repr(e)
        log.exception("%s: run failed", site.name)
    store.log_run(mode, site.name, started, stats)
    return stats


def run_once(settings: Settings, store: Store, sites: list[Site], **kwargs) -> int:
    """One pass: scrape all given sites in parallel, print a summary table, run health checks.

    Used by `run`, `backfill` and every tick of `loop`. One thread per site; extra kwargs (mode, since, limit,
    include) are passed through to scrape_site. Returns 1 if any health check FAILs, else 0 (the exit code).
    """
    def scrape(site: Site) -> dict:
        return scrape_site(site, settings, store, **kwargs)

    with ThreadPoolExecutor(max_workers=len(sites) or 1) as pool:
        all_stats = pool.map(scrape, sites)          # same order as `sites`
        results = {site.name: stats for site, stats in zip(sites, all_stats)}

    print_run_summary(results)
    # Health covers all enabled sites, not just the ones scraped now: the phrase check compares outlets.
    return report_health(store, enabled(settings), results)


def catch_up(settings: Settings, store: Store, sites: list[Site]):
    """Before `loop` starts: backfill each site from its last fetch until now, so downtime leaves no holes.

    Feeds only hold their newest items, so anything published while the scraper was off may have dropped
    off them. For each site whose last fetch is over CATCHUP_MIN_GAP_HOURS old, this runs a backfill
    starting an hour before that fetch (safety margin), but no further back than `catchup_max_days`
    and at most `catchup_limit` articles. Sites never scraped before are skipped (the live run starts them).
    """
    starts: dict[str, str] = {}                      # site name -> backfill from this ISO time
    oldest_allowed = datetime.fromisoformat(days_ago_iso(settings.catchup_max_days))
    for site in sites:
        last = store.query("SELECT MAX(fetched_at) FROM articles WHERE source = ?", (site.name,))[0][0]
        if last is None or last >= days_ago_iso(CATCHUP_MIN_GAP_HOURS / 24):
            continue                                 # never scraped, or the feed still covers the gap
        if not site.backfill:
            log.warning("catch-up: %s has no backfill sources; articles since %s may be missing", site.name, last)
            continue
        since = max(datetime.fromisoformat(last) - timedelta(hours=1), oldest_allowed)
        starts[site.name] = since.isoformat(timespec="seconds")

    if not starts:
        log.info("catch-up: nothing to fill")
        return
    todo = [s for s in sites if s.name in starts]
    for s in todo:
        log.info("catch-up: %s from %s", s.name, starts[s.name])

    def fill(site: Site) -> dict:
        return scrape_site(site, settings, store, mode="backfill", since=starts[site.name],
                           limit=settings.catchup_limit)

    with ThreadPoolExecutor(max_workers=len(todo)) as pool:
        results = {site.name: stats for site, stats in zip(todo, pool.map(fill, todo))}
    print_run_summary(results)
    for name, stats in results.items():
        if stats.get("new", 0) >= settings.catchup_limit:     # gap bigger than the cap: older part still missing
            log.warning("catch-up: %s hit catchup_limit (%d); fill the rest with "
                        "`backfill --sites %s --days %g`", name, settings.catchup_limit, name, settings.catchup_max_days)


def print_run_summary(results: dict[str, dict]):
    """Print the per-site counts table after a run or catch-up."""
    columns = ["listed", "new", "fetched", "ok", "errors"]
    table = []
    for name, stats in results.items():
        table.append([name] + [stats.get(c, 0) for c in columns])
    print_table(["site"] + columns, table)


def report_health(store: Store, sites: list[Site], run_stats: dict | None = None) -> int:
    """Run health.check_all, save the findings in the `health` table, log every FAIL. Returns 1 if any FAIL."""
    findings = health.check_all(store, sites, run_stats)
    store.log_health(findings)
    failed = [f for f in findings if f.level == "FAIL"]
    for f in failed:
        log.error("HEALTH FAIL %s [%s] %s", f.source, f.check, f.message)
    if not failed:
        log.info("health: all checks passed")
    return 1 if failed else 0


# ── maintenance commands ─────────────────────────────────────────────────────

def reextract(store: Store, sites: list[Site], only_older: bool):
    """`reextract`: re-run extraction on the saved raw HTML, without downloading again.

    Used after changing extract.py or a site's strip/prune_xpath in sites.toml. Each row is overwritten in
    place, keeping its url, fetch time and raw file. Rows whose HTML was pruned are skipped.
    `only_older` limits it to rows extracted by an older EXTRACTOR_VERSION.
    """
    for site in sites:
        sql = "SELECT * FROM articles WHERE source = ? AND raw_path IS NOT NULL"
        params = [site.name]
        if only_older:
            sql += " AND extractor_version < ?"
            params.append(EXTRACTOR_VERSION)
        rows = store.query(sql, params)

        for old in rows:
            row = extract(store.load_raw(old["raw_path"]), old["url"], site, old["published_at"])
            row.update({k: old[k] for k in ("url", "source", "fetched_at", "http_status", "raw_path")})
            store.save_article(row)
        log.info("%s: re-extracted %d articles", site.name, len(rows))


def prune(store: Store, days: float):
    """`prune` (and once a day in `loop`): delete raw HTML older than `days`. Articles stay in the database."""
    files, freed = store.prune_raw(days)
    log.info("prune: deleted raw HTML of %d articles older than %g days (%.1f MB freed)", files, days, freed / 1e6)


def show_articles(store: Store, n: int, source: str | None, url: str | None, full: bool):
    """`show`: print the most recently fetched articles with all their fields, as the rest of the project sees them.

    Filters by outlet (`--source`) or one URL (`--url`); text is cut at 600 characters unless `--full`.
    """
    sql = "SELECT * FROM articles WHERE 1 = 1"      # always-true start, so each filter can just add "AND ..."
    params = []
    if source:
        sql += " AND source = ?"
        params.append(source)
    if url:
        sql += " AND url = ?"
        params.append(url)
    sql += " ORDER BY fetched_at DESC LIMIT ?"
    params.append(n)
    rows = store.query(sql, params)

    if not rows:
        print("no matching articles")
    for r in rows:
        print("─" * 80)
        for key in ("source", "url", "title", "published_at", "fetched_at", "section", "is_opinion",
                    "n_words", "ok", "dup_of", "raw_path"):
            print(f"{key:>13}: {r[key]}")
        text = r["text"] or ""
        if not full and len(text) > 600:
            text = text[:600] + " […]"
        print(f"{'text':>13}:\n\n{text}\n")


def stats_table(store: Store):
    """`stats`: one line per outlet with totals, usable articles, last 24h, opinion share, duplicates, length, dates."""
    rows = store.query("""
        SELECT source, COUNT(*) total, SUM(ok) ok, SUM(fetched_at >= ?) last_24h,
               ROUND(AVG(is_opinion), 2) opinion, SUM(dup_of IS NOT NULL) near_dups,
               CAST(AVG(CASE WHEN ok=1 THEN n_words END) AS INT) avg_words,
               MIN(published_at) first_published, MAX(published_at) last_published
        FROM articles GROUP BY source ORDER BY source""", (days_ago_iso(1),))
    print_table(rows[0].keys() if rows else ["source"], [list(r) for r in rows])


def export_jsonl(store: Store, out: Path, hours: float):
    """`export`: write recent usable articles to a JSONL file for the Kafka/Spark backend.

    Keeps the old scraper's format that backend/producer.py and spark_processor.py expect (source, url,
    title, date as epoch milliseconds, text), plus the new fields, which the Spark schema simply ignores.
    """
    rows = store.query("SELECT * FROM articles WHERE ok=1 AND fetched_at >= ? ORDER BY published_at",
                       (days_ago_iso(hours / 24),))
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            date = r["published_at"] or r["fetched_at"]
            f.write(json.dumps({
                "source": r["source"], "url": r["url"], "title": r["title"],
                "date": int(datetime.fromisoformat(date).timestamp() * 1000), "text": r["text"],
                "published_at": r["published_at"], "section": r["section"], "is_opinion": r["is_opinion"],
                "dup_of": r["dup_of"],
            }, ensure_ascii=False) + "\n")                              # ensure_ascii=False keeps Greek readable
    log.info("exported %d articles to %s", len(rows), out)


def print_table(header, rows):
    """Print rows as aligned text columns. Used by run summaries and `stats`."""
    lines = [[str(h) for h in header]]
    for row in rows:
        lines.append(["" if v is None else str(v) for v in row])

    # Each column is as wide as its longest cell.
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    for line in lines:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths)))


# ── CLI ──────────────────────────────────────────────────────────────────────

def enabled(settings: Settings) -> list[Site]:
    """Sites with enabled = true in sites.toml. The default set for every command and for health checks."""
    return [s for s in settings.sites if s.enabled]


def pick_sites(settings: Settings, names: str | None) -> list[Site]:
    """Turn `--sites a.gr,b.gr` into Site objects (all enabled sites if not given). Exits on unknown names."""
    if not names:
        return enabled(settings)
    wanted = set(names.split(","))
    unknown = wanted - {s.name for s in settings.sites}
    if unknown:
        sys.exit(f"unknown site(s): {', '.join(sorted(unknown))} (see sites.toml)")
    return [s for s in settings.sites if s.name in wanted]          # explicit names may be disabled ones


def main(argv=None) -> int:
    """Entry point for `python -m live_scraper ...` (called by __main__.py). Returns the process exit code.

    Defines the commands and their options, sets up logging, loads sites.toml, opens the database,
    then runs the chosen command.
    """
    # Commands and their options.
    parser = argparse.ArgumentParser(prog="python -m live_scraper", description="HellenicInsight news scraper")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "loop", "backfill", "reextract"):
        p = sub.add_parser(name)
        p.add_argument("--sites", help="comma-separated site names (default: all enabled)")
        if name == "run":
            p.add_argument("--limit", type=int, help="max new articles per site (default: max_new_per_run)")
        if name == "backfill":
            p.add_argument("--days", type=float, default=30, help="ignore articles older than this")
            p.add_argument("--limit", type=int, default=300, help="max new articles per site")
            p.add_argument("--include", help="regex the article URL must match, e.g. '/(politiki|politics)/'")
        if name == "reextract":
            p.add_argument("--only-older", action="store_true", help="only rows from an older EXTRACTOR_VERSION")
    p = sub.add_parser("prune")
    p.add_argument("--raw-days", type=float, help="keep raw HTML this many days (default: raw_keep_days in sites.toml)")
    sub.add_parser("health")
    sub.add_parser("stats")
    p = sub.add_parser("show")
    p.add_argument("-n", type=int, default=1, help="how many of the most recently fetched articles")
    p.add_argument("--source", help="only this outlet, e.g. efsyn.gr")
    p.add_argument("--url", help="one specific article")
    p.add_argument("--full", action="store_true", help="print the whole text, not the first 600 characters")
    p = sub.add_parser("export")
    p.add_argument("--out", type=Path, default=PACKAGE_DIR / "kafka_feed.jsonl")
    p.add_argument("--hours", type=float, default=24)
    args = parser.parse_args(argv)

    # Logging: our messages at INFO (DEBUG with -v); the libraries' own chatter only on errors.
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("trafilatura", "htmldate", "urllib3", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    settings = load_settings()
    store = Store(settings.data_dir)
    sites = pick_sites(settings, getattr(args, "sites", None))      # commands without --sites get all enabled

    # Dispatch to the chosen command.
    if args.command == "run":
        return run_once(settings, store, sites, limit=args.limit)
    if args.command == "loop":
        try:
            catch_up(settings, store, sites)                            # fill the gap since the last run
        except Exception:
            log.exception("catch-up failed; starting the loop anyway")
        last_prune = 0.0
        while True:                                                     # never dies: errors are logged, next tick runs
            started = time.monotonic()
            try:
                run_once(settings, store, sites)
                if started - last_prune > 24 * 3600 or not last_prune:  # once at start, then daily
                    prune(store, settings.raw_keep_days)
                    last_prune = started
            except Exception:
                log.exception("run failed")
            # Sleep the rest of the interval: a 5-minute run with loop_minutes = 15 sleeps 10 minutes.
            time.sleep(max(0.0, settings.loop_minutes * 60 - (time.monotonic() - started)))
    if args.command == "backfill":
        return run_once(settings, store, sites, mode="backfill", since=days_ago_iso(args.days),
                        limit=args.limit, include=args.include)
    if args.command == "reextract":
        reextract(store, sites, args.only_older)
        return report_health(store, enabled(settings))
    if args.command == "prune":
        prune(store, settings.raw_keep_days if args.raw_days is None else args.raw_days)
    if args.command == "health":
        return report_health(store, enabled(settings))
    if args.command == "stats":
        stats_table(store)
    if args.command == "show":
        show_articles(store, args.n, args.source, args.url, args.full)
    if args.command == "export":
        export_jsonl(store, args.out, args.hours)
    return 0
