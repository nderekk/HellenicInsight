"""Storage: one SQLite file for rows + gzipped raw HTML files on disk.

Everything the scraper keeps goes through Store: articles, raw HTML, the run log and health results.
  data/scraper.db                                   tables: articles, runs, health
  data/raw/<source>/<YYYY-MM>/<sha1(url)>.html.gz   the page as downloaded; the article row stores this path
Raw HTML is in files (not the DB) because it's ~6x bigger than the extracted row and is deleted after 30 days.
"""
import gzip
import hashlib
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Created on first run; "IF NOT EXISTS" makes it safe to run every time.
SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    url               TEXT PRIMARY KEY,  -- normalized URL = the article's identity
    source            TEXT NOT NULL,     -- outlet, e.g. "efsyn.gr"
    title             TEXT,
    author            TEXT,
    published_at      TEXT,            -- ISO UTC
    fetched_at        TEXT NOT NULL,   -- ISO UTC
    section           TEXT,
    is_opinion        INTEGER,
    text              TEXT,
    n_words           INTEGER,
    ok                INTEGER,         -- 1 if n_words >= site.min_words
    text_sha1         TEXT,            -- exact-duplicate fingerprint
    simhash           INTEGER,         -- near-duplicate fingerprint; signed 64-bit (SQLite has no unsigned ints)
    sh_b0 INTEGER, sh_b1 INTEGER, sh_b2 INTEGER, sh_b3 INTEGER,   -- simhash split in 4 x 16 bits, indexed for fast lookup
    dup_of            TEXT,            -- url of an earlier near-identical article, if any
    raw_path          TEXT,            -- raw HTML file; NULL once pruned
    http_status       INTEGER,
    extractor_version INTEGER          -- extract.EXTRACTOR_VERSION that produced this row
);
-- Indexes = faster lookups on these columns (per-outlet health queries, date filters, duplicate search).
CREATE INDEX IF NOT EXISTS idx_articles_source_fetched ON articles(source, fetched_at);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_b0 ON articles(sh_b0);
CREATE INDEX IF NOT EXISTS idx_b1 ON articles(sh_b1);
CREATE INDEX IF NOT EXISTS idx_b2 ON articles(sh_b2);
CREATE INDEX IF NOT EXISTS idx_b3 ON articles(sh_b3);

-- One row per site per run: counts shown in the run summary.
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT, source TEXT, started_at TEXT, finished_at TEXT,
    listed INTEGER, new INTEGER, fetched INTEGER, ok INTEGER, errors INTEGER, error TEXT
);

-- Every health check result (OK and FAIL), for looking back at when something broke.
CREATE TABLE IF NOT EXISTS health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at TEXT, source TEXT, check_name TEXT, level TEXT, message TEXT
);
"""

# Near-duplicates: texts whose 64-bit simhashes differ in <= 3 bits. Split into 4 bands of 16 bits, two such
# hashes must share at least one band exactly (3 differences can touch at most 3 bands), so only articles
# sharing a band need checking, and that's an indexed lookup.
NEAR_DUP_BITS = 3
NEAR_DUP_WINDOW_DAYS = 7     # only compare with articles from the last week (wire stories repeat within days)


def now_iso() -> str:
    """Current time as ISO UTC text, same format as discover.to_utc_iso."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def days_ago_iso(days: float) -> str:
    """ISO UTC text for `days` ago. Used for "since" filters, e.g. fetched_at >= days_ago_iso(1)."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


def _signed(h: int) -> int:
    """Unsigned 64-bit -> signed, because SQLite integers are signed (values >= 2^63 would overflow)."""
    return h - (1 << 64) if h >= 1 << 63 else h


class Store:
    """The database connection plus the raw HTML folder. One Store per process, shared by all site threads."""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        (self.dir / "raw").mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: site threads share this connection (self.lock keeps it safe).
        # isolation_level=None: autocommit, every statement is saved immediately.
        self.db = sqlite3.connect(self.dir / "scraper.db", check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row   # rows readable by column name: row["title"]
        self.db.execute("PRAGMA journal_mode=WAL")   # lets you read the DB (e.g. a notebook) while the scraper writes
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()        # one thread uses the connection at a time

    # ── articles ─────────────────────────────────────────────────────────────

    def known_urls(self, urls) -> set[str]:
        """Which of these URLs are already stored. Used to skip articles we already have."""
        urls = list(urls)
        found = set()
        with self.lock:
            # In chunks of 500: SQLite limits how many "?" placeholders one query may have.
            for i in range(0, len(urls), 500):
                chunk = urls[i:i + 500]
                placeholders = ",".join("?" * len(chunk))
                rows = self.db.execute(f"SELECT url FROM articles WHERE url IN ({placeholders})", chunk)
                found.update(r[0] for r in rows)
        return found

    def save_raw(self, source: str, url: str, html: bytes) -> str:
        """Save a downloaded page gzipped. Returns its path relative to data_dir (stored in the row's raw_path).

        File name = SHA-1 of the URL: unique per article and safe as a file name. Grouped by month for pruning.
        """
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        name = hashlib.sha1(url.encode()).hexdigest() + ".html.gz"
        rel = Path("raw") / source / month / name
        (self.dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.dir / rel).write_bytes(gzip.compress(html))
        return str(rel)

    def load_raw(self, rel: str) -> bytes:
        """Read back a saved page. Used by `reextract`."""
        return gzip.decompress((self.dir / rel).read_bytes())

    def prune_raw(self, days: float) -> tuple[int, int]:
        """Delete raw HTML fetched more than `days` ago. Rows and text stay; only `reextract` loses them.

        Returns (articles pruned, bytes freed). Used by `prune` and daily by `loop`.
        """
        rows = self.query("SELECT url, raw_path FROM articles WHERE raw_path IS NOT NULL AND fetched_at < ?",
                          (days_ago_iso(days),))
        freed = 0
        for r in rows:
            path = self.dir / r["raw_path"]
            if path.exists():
                freed += path.stat().st_size
                path.unlink()
        with self.lock:
            self.db.executemany("UPDATE articles SET raw_path = NULL WHERE url = ?", [(r["url"],) for r in rows])
        for month_dir in (self.dir / "raw").glob("*/*"):          # drop emptied raw/<source>/<YYYY-MM> folders
            if month_dir.is_dir() and not any(month_dir.iterdir()):
                month_dir.rmdir()
        return len(rows), freed

    def save_article(self, row: dict):
        """Insert one article, or replace it if the URL exists (that's how `reextract` updates rows).

        `row` comes from extract.extract plus url/source/fetched_at/raw_path. Before saving, the simhash is
        split into bands and dup_of is set if an earlier near-identical text was stored in the last week.
        """
        h = row.pop("simhash", None)
        if h is not None and row.get("text"):
            bands = [(h >> (16 * i)) & 0xFFFF for i in range(4)]     # the 4 x 16-bit pieces of the hash
            row.update(simhash=_signed(h), sh_b0=bands[0], sh_b1=bands[1], sh_b2=bands[2], sh_b3=bands[3])
            row["dup_of"] = self._find_near_dup(row["url"], row["fetched_at"], h, bands)

        columns = ", ".join(row)
        placeholders = ", ".join("?" * len(row))
        with self.lock:
            self.db.execute(f"INSERT OR REPLACE INTO articles ({columns}) VALUES ({placeholders})",
                            list(row.values()))

    def _find_near_dup(self, url: str, fetched_at: str, h: int, bands: list[int]) -> str | None:
        """URL of an earlier article whose simhash is within NEAR_DUP_BITS of `h`, or None.

        Only articles fetched before this one count, so the link always points to the first copy
        (e.g. an ΑΠΕ-ΜΠΕ story on efsyn and then iefimerida: the iefimerida row gets dup_of = the efsyn url).
        Two threads can fetch copies in the same second; then the smaller URL counts as first, so exactly
        one of the pair gets the link.
        """
        with self.lock:
            candidates = self.db.execute(
                "SELECT url, simhash FROM articles WHERE (sh_b0=? OR sh_b1=? OR sh_b2=? OR sh_b3=?) "
                "AND (fetched_at < ? OR (fetched_at = ? AND url < ?)) AND fetched_at >= ? "
                "ORDER BY fetched_at LIMIT 200",
                (*bands, fetched_at, fetched_at, url, days_ago_iso(NEAR_DUP_WINDOW_DAYS))).fetchall()
        for c in candidates:
            stored = c["simhash"] & 0xFFFFFFFFFFFFFFFF                # back to unsigned
            differing_bits = bin(stored ^ h).count("1")              # XOR marks bits that differ
            if differing_bits <= NEAR_DUP_BITS:
                return c["url"]
        return None

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        """Run a read query safely from any thread. Used by health checks and the commands."""
        with self.lock:
            return self.db.execute(sql, params).fetchall()

    # ── runs & health log ────────────────────────────────────────────────────

    def log_run(self, mode: str, source: str, started_at: str, stats: dict):
        """Record one site's run (counts from scraper.scrape_site) in the `runs` table."""
        with self.lock:
            self.db.execute(
                "INSERT INTO runs (mode, source, started_at, finished_at, listed, new, fetched, ok, errors, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mode, source, started_at, now_iso(), stats.get("listed", 0), stats.get("new", 0),
                 stats.get("fetched", 0), stats.get("ok", 0), stats.get("errors", 0), stats.get("error")))

    def log_health(self, findings):
        """Record health check results (health.Finding objects) in the `health` table."""
        with self.lock:
            self.db.executemany(
                "INSERT INTO health (checked_at, source, check_name, level, message) VALUES (?, ?, ?, ?, ?)",
                [(now_iso(), f.source, f.check, f.level, f.message) for f in findings])
