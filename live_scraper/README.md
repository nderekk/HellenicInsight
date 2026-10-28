# live_scraper — how we collect articles

Every 15 minutes, the scraper asks each Greek news site for its list of newest articles, downloads the new ones, extracts the article text, and stores it in a local database. It also checks itself after every run and fails loudly when a site breaks.

- [Quick start](#quick-start)
- [Key ideas: feeds, sitemaps, and why we don't crawl](#key-ideas-feeds-sitemaps-and-why-we-dont-crawl)
- [The flow, step by step](#the-flow-step-by-step)
- [The flow in the code: one loop, function by function](#the-flow-in-the-code-one-loop-function-by-function)
- [Where the data ends up](#where-the-data-ends-up)
- [Being polite (and what that does not cover)](#being-polite-and-what-that-does-not-cover)
- [Health checks](#health-checks)
- [Adding or fixing an outlet](#adding-or-fixing-an-outlet)
- [Files and who uses what](#files-and-who-uses-what)
- [All commands](#all-commands)

---

## Quick start

From the repo root, with the virtual environment active:

```bash
pip install -r live_scraper/requirements.txt

python -m live_scraper run        # one pass over all outlets (~5 min), then health checks
python -m live_scraper stats      # how many articles per outlet
python -m live_scraper loop       # fill the gap since the last run, then scrape every 15 min (Ctrl+C to stop)
```

To leave it running in the background and log to a file:

```bash
nohup python -m live_scraper loop > live_scraper/data/loop.log 2>&1 &
```

---

## Key ideas: feeds, sitemaps, and why we don't crawl

**Feed (RSS / Atom).** A file a news site publishes so that programs can see its latest articles without opening its pages. It is plain XML, with one `<item>` per article:

```xml
<item>
  <title>Τέμπη: Συγκλονίζουν οι μαρτυρίες των επιζώντων…</title>
  <link>https://www.efsyn.gr/ellada/1496733/tempi-sygklonizoyn-…/</link>
  <pubDate>Mon, 26 Oct 2026 10:07:33 +0000</pubDate>
  <description>Συγκλονιστικές μαρτυρίες επιζώντων…</description>
</item>
```

A feed only holds the **newest** items (10 for thepressproject, 90 for efsyn, 100 for protothema). Old ones drop off as new ones arrive, which is why we poll every 15 minutes.

**Sitemap.** A file a site publishes **for search engines**, listing its pages. It is simpler XML: mostly URLs, sometimes with a date. Big sites split it into many files under a **sitemap index** (protothema has 78 files of about 480 articles each). Sitemaps reach much further back than feeds, so we use them to collect older articles (`backfill`).

**Crawling, and why we don't do it.** A crawler starts from a page, follows every link on it, then every link on those pages, and so on. It explores the whole site, downloads many pages that aren't articles (menus, tags, archives), and has to guess what is an article. We do something narrower: we **ask the site for its own list** (feed or sitemap) and download **only the URLs on that list**. We never follow links found inside articles. That means:

- far fewer requests (politer, and less likely to get us blocked);
- no guessing what is an article;
- much less breakage, because feed formats are standard and rarely change, while page layouts change often.

**Why we only take the URL and date from a feed.** The feed's `<description>` is usually a teaser, not the full article; the full text is only on the article page. So from the feed we take:

- the **URL**, to know which page to download (and to skip ones we already have);
- the **date**, as a fallback publish time if the page doesn't state one. In `backfill` it also tells us when to stop going back in time.

The feed itself is then thrown away.

---

## The flow, step by step

```
                      sites.toml (which outlets, their feeds, cleanup rules)
                                  │
                                  ▼
   ┌──────────── one thread per outlet, all running at the same time ────────────┐
   │                                                                              │
   │  1. LIST      download feed/sitemap ──► article URLs + dates                 │
   │                                       (the feed itself is discarded)         │
   │  2. FILTER    clean URLs, keep this outlet's links, drop ones already in DB  │
   │  3. FETCH     download each new article page (3 s apart, robots.txt obeyed)  │
   │  4. SAVE RAW  store the page as compressed HTML on disk                      │
   │  5. EXTRACT   HTML ──► title, text, date, section, is_opinion, hash          │
   │  6. STORE     one row per article in the database (+ duplicate check)        │
   │                                                                              │
   └──────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
               7. HEALTH CHECKS on the whole database ──► FAIL = loud error + exit code 1
```

1. **List** (`discover.py`). Download each URL in the outlet's `feeds` and read out the article URLs and dates. The format (RSS, Atom, sitemap, sitemap index) is detected automatically.
2. **Filter** (`scraper.py`). Remove tracking junk from URLs (`?utm_source=rss`), so one article always has one URL. Keep only links to this outlet, apply the outlet's include/exclude rules, and skip URLs already in the database.
3. **Fetch** (`net.py`). Download each new article page, one at a time per outlet, with at least 3 seconds between requests to the same site.
4. **Save raw** (`store.py`). Save the downloaded page, unchanged, as a compressed file. If extraction turns out wrong, we can fix it and re-extract from this file without downloading again (`reextract`).
5. **Extract** (`extract.py`). [trafilatura](https://trafilatura.readthedocs.io/) finds the main article text and drops menus, ads and related links. We then apply the outlet's own cleanup rules from `sites.toml` and work out the publish date, section (from the URL), whether it's an opinion piece, and a fingerprint hash.
6. **Store** (`store.py`). Write one row per article to the database. If an almost identical text arrived in the last week (e.g. the same ΑΠΕ-ΜΠΕ wire story on several sites), the row is linked to it in `dup_of`.
7. **Health** (`health.py`). After all outlets finish, check the whole database for signs that something broke (see [Health checks](#health-checks)).

**Threads.** Everything runs **on your computer**, inside the one Python process you start. `run_once` in `scraper.py` starts **one thread per outlet** (5 outlets = 5 threads). Inside a thread, that outlet's articles are downloaded one after another. Threads help because almost all the time is spent *waiting*, for the network or the 3-second politeness delay, and 5 outlets can wait at the same time. A run takes about as long as the slowest outlet (~5 min) instead of all of them added up (~15–20 min). Each outlet has its own rate limit, so efsyn never waits for protothema, but never gets more than one request every 3 seconds from us. All threads share one database connection, and a lock in `store.py` lets only one write at a time.

---

## The flow in the code: one loop, function by function

This follows `python -m live_scraper loop` from the moment you press Enter. `run` is the same, but skips the catch-up (step 2b) and stops after step 7. Every step names **file → function**, so you can open it and read along. All files are in `live_scraper/`.

**Startup (once)**

1. **`__main__.py`** runs when you type `python -m live_scraper`. It only calls **`scraper.py → main()`**.
2. **`scraper.py → main()`** reads the command you typed (`loop`) with `argparse`. Then it calls three things:
   - **`scraper.py → load_settings()`** reads `sites.toml`. It turns each `[[site]]` block into a **`Site`** object (one outlet's settings) inside a **`Settings`** object (global settings + all sites). Both classes are defined at the top of `scraper.py`.
   - **`store.py → Store(...)`** opens `data/scraper.db`, creating the tables the first time. This `store` object is how every later step reads and writes data.
   - **`scraper.py → pick_sites()`** chooses which outlets to scrape: all with `enabled = true`, or the ones named with `--sites`.

   **2b. Catch-up (`loop` only).** **`scraper.py → catch_up()`** fills any gap left while the scraper was off. Feeds only hold their newest items, so articles published during downtime may have dropped off them. For each outlet whose last fetch (from the database) is more than 1 hour old, it runs the same `scrape_site()` as step 5, in backfill mode (older pages/sitemaps instead of the feed). It starts 1 hour before that last fetch, and is capped by `catchup_max_days` (3) and `catchup_limit` (500 articles) in `sites.toml`. If an outlet hits the cap, a warning says how to fill the rest with `backfill`.

**One run (repeats every 15 minutes)**

3. **`scraper.py → main()`**, in its `loop` branch, calls **`scraper.py → run_once()`**.
4. **`scraper.py → run_once()`** starts one thread per outlet. Each thread runs **`scraper.py → scrape_site()`** for its outlet. Steps 5a–5e happen inside every thread, for its own outlet, all at the same time.

**Inside `scrape_site()` for one outlet**

5. **`scraper.py → scrape_site()`** does the whole job for one outlet:
   - **a. Get a downloader.** It creates a **`net.py → Fetcher`** for this outlet. All downloads in steps b and d go through its **`Fetcher.get()`**, which before each request:
     - checks the site's robots.txt with **`Fetcher.allowed()`**. The first time, it downloads and reads robots.txt with **`Fetcher._load_robots()`**;
     - waits until 3 s have passed since the previous request, in **`Fetcher._get()`**;
     - returns `None` if robots.txt forbids the URL. Otherwise it returns the response.
   - **b. List the articles.** For each URL in the outlet's `feeds`, it calls **`scraper.py → read_listing()`**:
     - **`scraper.py → fetch_listing()`** downloads the feed with `Fetcher.get()` and hands the bytes to **`discover.py → parse_listing()`**. That function detects RSS/Atom/sitemap and returns the article URLs with their dates.
     - For a sitemap index, `read_listing()` also reads its 2 newest sub-sitemaps the same way.
     - Each URL is cleaned by **`discover.py → normalize_url()`** (tracking junk such as `?utm_source=rss` is removed), so one article always has one URL.
   - **c. Keep only what's new.**
     - **`scraper.py → keep()`** drops links to other domains and URLs matching the outlet's exclude rules.
     - **`store.py → Store.known_urls()`** finds which URLs are already in the database; those are dropped.
     - **`scraper.py → newest_first()`** sorts the rest newest first (undated ones by the article ID in the URL, via **`discover.py → last_number()`**).
     - The list is capped at `max_new_per_run` (200).
   - **d. For each new article URL:**
     1. **Download** it with `Fetcher.get()`. If that returns `None` or a status other than 200, count an error and move to the next URL.
     2. **Extract** it with **`extract.py → extract()`**, which turns the HTML into the article's fields:
        - `trafilatura.bare_extraction()` (an external library) finds the main text;
        - the outlet's `strip` patterns from `sites.toml` remove leftovers;
        - **`_tidy_whitespace()`** cleans spacing;
        - **`_meta_published()`** reads the publish date from the page. If it finds none, the feed's date is used; both are converted by **`discover.py → to_utc_iso()`**;
        - **`_clean_title()`**, **`_section()`** and **`_is_opinion()`** fill in title, section and opinion flag;
        - **`simhash()`** computes the near-duplicate fingerprint.
        All of these helpers are in `extract.py`. The result is a dictionary: one database row, missing only url/source/fetch time.
     3. **Save the raw page** with **`store.py → Store.save_raw()`**. It writes `data/raw/<outlet>/<YYYY-MM>/<hash>.html.gz` and returns that path, which goes into the row as `raw_path`.
     4. **Save the row** with **`store.py → Store.save_article()`**. Before inserting, it calls **`Store._find_near_dup()`**: if an earlier article in the last 7 days has an almost identical text, the row gets `dup_of` = that article's URL.

     One article failing (e.g. a timeout) is logged and skipped; it doesn't stop the outlet.
   - **e. Log the run.** **`store.py → Store.log_run()`** writes this outlet's counts (listed, new, fetched, ok, errors) to the `runs` table. `scrape_site()` returns those counts to `run_once()`.

**After all outlets finish (back in `run_once()`)**

6. **`scraper.py → print_run_summary()`** prints the summary table (`site  listed  new  fetched  ok  errors`), using **`scraper.py → print_table()`** to align the columns.
7. **`scraper.py → report_health()`** calls **`health.py → check_all()`**, which runs:
   - **`check_volume()`** and **`check_length()`** for each outlet;
   - **`check_phrases()`** once, comparing all outlets with each other.

   All three live in `health.py`. The results are saved by **`store.py → Store.log_health()`**. Each FAIL is logged as `HEALTH FAIL …`, and `run_once()` returns 1 if anything failed (for `run`, that becomes the exit code).

**Back in the loop (`scraper.py → main()`)**

8. After the first run, and then once a day, **`scraper.py → prune()`** calls **`store.py → Store.prune_raw()`**, which deletes raw HTML older than 30 days.
9. `main()` sleeps for the rest of the 15 minutes (a 5-minute run sleeps 10 minutes), then **goes back to step 3**. An error in a run is logged and the loop continues; only Ctrl+C (or killing the process) stops it.

The same flow on one screen. Numbers match the steps above, and every name is `file.function` (e.g. `store.save_raw()` = function `save_raw` in `store.py`):

```
  python -m live_scraper loop
     │
     ▼
 1–2 __main__.py → scraper.main()
       ├ scraper.load_settings()   read sites.toml → Settings + Sites
       ├ store.Store()             open data/scraper.db
       ├ scraper.pick_sites()      the enabled outlets
       └ scraper.catch_up()        (2b) backfill the gap since last fetch
     │
     ▼
 3–4 scraper.run_once()                                         ◄──┐
     │  one thread per outlet, each runs:                          │
     ▼                                                             │
 5   scraper.scrape_site(outlet)                                   │
       a  net.Fetcher()             polite downloader              │
       b  scraper.read_listing()    feed(s) via Fetcher.get()      │
            → discover.parse_listing()   feed → URLs + dates       │
            → discover.normalize_url()   one URL per article       │
       c  scraper.keep()            this outlet's links only       │
          store.known_urls()        drop ones already stored       │
          scraper.newest_first()    newest first, max 200          │
       d  for each new URL:                                        │
            net.Fetcher.get()       download the page              │
            extract.extract()       HTML → title, text, date…      │
            store.save_raw()        raw HTML → data/raw/…          │
            store.save_article()    row → DB (+ dup check)         │
       e  store.log_run()           counts → runs table            │
     │  (waits for all outlets to finish)                          │
     ▼                                                             │
 6   scraper.print_run_summary()    summary table                  │
 7   scraper.report_health()                                       │
       → health.check_all()         volume, length, phrases        │
       → store.log_health()         results → health table         │
     │                                                             │
     ▼                                                             │
 8   scraper.prune() → store.prune_raw()     once a day            │
 9   sleep the rest of 15 min ─────────────────────────────────────┘
```

---

## Where the data ends up

Everything lives in `live_scraper/data/`, which is **not in git** (it's local to each machine).

| What | Where | Kept | Used by |
|---|---|---|---|
| **Extracted article** (title, text, date, section, …) | `data/scraper.db`, table `articles`, one row per article | forever | everything: research, health, stats, export |
| **Raw HTML** (the page as downloaded) | `data/raw/<outlet>/<YYYY-MM>/<hash>.html.gz`; the row's `raw_path` points to it | 30 days (`prune`) | `reextract` only |
| Run log and health results | `data/scraper.db`, tables `runs` and `health` | forever | debugging |
| Export for the Kafka/Spark backend | `live_scraper/kafka_feed.jsonl` | overwritten on each `export` | `backend/producer.py` |

- **Why the raw HTML is in files, not the database.** It's about 42 KB per article against about 7 KB for the extracted row. Files keep the database small and fast, and old HTML can be deleted (`prune`) without touching the articles. The file name is a hash of the article's URL, so every URL gets a unique, safe name.
- **Is it extracted from the saved file?** No. Extraction runs on the downloaded page in memory, at the same moment the file is saved. The file is only read again by `reextract`, which overwrites the same database row (it doesn't make a new copy).
- **Storage use.** About 1,000–1,500 articles/day across 5 outlets. Raw HTML adds ~50 MB/day but is capped at ~1.5 GB by the 30-day prune (`loop` prunes daily). The database grows ~9 MB/day, about 3.5 GB/year.

**Reading the data.** In Python or a notebook:

```python
import sqlite3, pandas as pd
df = pd.read_sql("SELECT * FROM articles WHERE ok = 1", sqlite3.connect("live_scraper/data/scraper.db"))
```

Main columns: `source`, `url`, `title`, `published_at` (UTC), `section`, `is_opinion`, `text`, `n_words`, `ok` (text long enough to use), `dup_of` (URL of an earlier near-identical article). [DB Browser for SQLite](https://sqlitebrowser.org/) also works if you prefer a GUI.

---

## Being polite (and what that does not cover)

- **robots.txt.** Every site publishes rules for bots at `/robots.txt`. We read them before downloading anything from a site and skip disallowed URLs. If it asks for a `Crawl-delay`, we use it when it's longer than ours.
- **Rate limit.** At least 3 seconds between our requests to the same site, and retries back off when a site says it's busy (429 / 5xx).
- **Identification.** Our requests say who we are: `HellenicInsightBot/0.1 (research; +https://github.com/nderekk/HellenicInsight)`.
- **Sites that block bots stay out.** avgi.gr and documentonews.gr answer article requests with a Cloudflare bot challenge, and kathimerini.gr's robots.txt forbids its feed and blocks AI crawlers by name. We don't work around any of that; the reasons are listed at the bottom of `sites.toml`.

**What this does not cover.** robots.txt is a technical convention, not a site's legal terms of use, and storing full article text raises copyright questions (EU press publishers' right, Greek Law 4996/2022). Our rule, from the roadmap (Phase 11): **full text stays internal, for analysis only. Publicly we show only title, a short snippet and a link to the original.** Check this before any public launch.

---

## Health checks

They run after every `run`, `loop` pass and `reextract`. Any FAIL is logged as `HEALTH FAIL …`, and `run` exits with code 1 so a scheduler can notice.

| Check | Fails when | Usual cause |
|---|---|---|
| volume | a feed lists 0 articles, every download of a run gave no usable text, or no new usable article for 12 h | feed URL moved, site blocks us, layout change |
| length | median article length over the last 24 h fell below half of the previous two weeks' | extraction now grabs only part of the page |
| phrases | a phrase is in ≥ 20% of one outlet's recent articles but ≤ 2% of the other outlets' | leftover page text: buttons, "read more" links, footers |

The phrase check matters most for the research side. Text that only one outlet has (like protothema's pop-up "Κλείσιμο" button, found in 40% of the old hackathon data) lets a model recognise the **outlet** instead of the article's **stance**. Treat a FAIL as a to-do: strip the phrase in `sites.toml`, or accept it with `allow_phrases` if it's real content (like ertnews's "Πηγή: ΑΠΕ-ΜΠΕ" wire credits).

---

## Adding or fixing an outlet

All per-outlet settings live in `sites.toml`; the code is the same for every site.

**Add one:**

1. Find its feed: try `https://www.<site>/feed/`, `/rss`, `/rss.xml`, or look for `application/rss+xml` in the homepage source. Check its `/robots.txt` too.
2. Copy a `[[site]]` block in `sites.toml`, then set `name` and `feeds` (and `backfill` if it has a paged feed or sitemap).
3. Run `python -m live_scraper run --sites <name>`, then `python -m live_scraper stats`.
4. Look at a few articles. If leftover text shows up, add `strip` patterns.

**Fix one** (e.g. after a `phrases` FAIL):

1. Add a `strip` regex (or `prune_xpath`) to the site's block in `sites.toml`.
2. Run `python -m live_scraper reextract --sites <name>`. This re-extracts from the saved HTML, so nothing is downloaded again.
3. Health runs automatically at the end of `reextract`; check that it passes.

---

## Files and who uses what

```
__main__.py ──► scraper.py ──reads──► sites.toml
                    │
                    ├──► net.py        downloading: robots.txt, delay, retries
                    ├──► discover.py   feed/sitemap ──► URLs + dates
                    ├──► extract.py ──► discover.py
                    ├──► store.py      database + raw HTML files
                    └──► health.py  ──► store.py
```

| File | Job |
|---|---|
| `sites.toml` | Which outlets, their feeds, per-site cleanup, global settings |
| `scraper.py` | The commands and the per-outlet flow (steps 1–7 above) |
| `net.py` | Polite downloading for one site |
| `discover.py` | Parses feeds and sitemaps |
| `extract.py` | HTML ──► article fields, plus the duplicate fingerprint |
| `store.py` | SQLite database, raw files, duplicate lookup, pruning |
| `health.py` | The three checks |

`net.py`, `discover.py` and `store.py` don't depend on any other project file, so each can be changed or replaced on its own.

---

## All commands

Run from the repo root as `python -m live_scraper <command>`. Add `--sites a.gr,b.gr` to `run`, `loop`, `backfill` or `reextract` to limit them to some outlets.

| Command | What it does |
|---|---|
| `run` | One pass over all enabled outlets, then health checks |
| `loop` | First fills the gap since the last fetch (catch-up), then `run` every 15 minutes forever; prunes raw HTML once a day |
| `backfill --days 30 --limit 300 --include "/(politiki\|politics)/"` | Older articles from paged feeds/sitemaps (here: politics only, last 30 days, max 300 per outlet) |
| `reextract` | Re-run extraction on saved HTML after changing `sites.toml` or `extract.py` |
| `prune --raw-days 30` | Delete raw HTML older than 30 days (articles stay) |
| `health` | Run the health checks only |
| `stats` | Articles per outlet: totals, last 24 h, average length, date range |
| `export --hours 24` | Write recent articles to `kafka_feed.jsonl` for the Kafka/Spark backend |
