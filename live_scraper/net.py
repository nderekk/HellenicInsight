"""Polite HTTP for one site: robots.txt, a minimum gap between requests, retries with backoff.

The only file that talks to the internet: feeds, sitemaps, robots.txt and articles all go through Fetcher.get.
Each site gets its own Fetcher (and runs in its own thread), so one site's delay never slows another.
"""
import logging
import time
from urllib import robotparser  # built-in robots.txt reader
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter  # lets a requests Session retry automatically
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


class Fetcher:
    """Downloads URLs for one site. Created per site per run by scraper.scrape_site."""

    def __init__(self, user_agent: str, delay_seconds: float, timeout: float = 20):
        self.user_agent = user_agent             # identifies us in every request (sites.toml [settings])
        self.delay = delay_seconds               # min seconds between requests; raised by robots.txt Crawl-delay
        self.timeout = timeout                   # give up on a request after this many seconds
        self._next_request_at = 0.0              # earliest moment (monotonic clock) the next request may start
        self._robots: dict[str, robotparser.RobotFileParser] = {}   # host -> parsed robots.txt, loaded once

        # A Session reuses the connection between requests: faster for us, lighter for the site.
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "el-GR,el;q=0.9,en;q=0.5"})
        # Retry only temporary errors (429 too many requests, 5xx server errors): up to 3 times, waiting
        # 0 s, 4 s, 8 s, or as long as the site's Retry-After header asks. Other responses (e.g. 404) are returned.
        retry = Retry(total=3, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",), respect_retry_after_header=True)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url: str) -> requests.Response | None:
        """Download a URL, respecting robots.txt and the delay. Used for every request the scraper makes.

        Returns None if robots.txt forbids the URL (the caller skips it); otherwise the response,
        whose status_code the caller checks.
        """
        if not self.allowed(url):
            log.warning("robots.txt disallows %s", url)
            return None
        return self._get(url)

    def allowed(self, url: str) -> bool:
        """True if robots.txt lets our bot fetch this URL. robots.txt is downloaded once per host and cached."""
        host = urlsplit(url).netloc
        if host not in self._robots:
            self._robots[host] = self._load_robots(host)
        return self._robots[host].can_fetch(self.user_agent, url)

    def _load_robots(self, host: str) -> robotparser.RobotFileParser:
        """Download and parse a host's robots.txt. Also applies its Crawl-delay if longer than ours.

        Uses our own session (robotparser's built-in downloader gets 403s on some sites) and _get, not get,
        so fetching robots.txt doesn't first check robots.txt.
        Known gap: Python's robotparser ignores "*" wildcards in paths (e.g. "Disallow: /*/media/oembed").
        """
        rp = robotparser.RobotFileParser()
        try:
            r = self._get(f"https://{host}/robots.txt")
        except requests.RequestException:
            r = None
        # Standard reading: 4xx = no robots.txt, allow all; 5xx or unreachable = rules unknown, allow nothing.
        # Unreachable blocks the site until the next run, which creates a new Fetcher.
        if r is None or r.status_code >= 500:
            rp.disallow_all = True
        elif r.status_code >= 400:
            rp.allow_all = True
        else:
            rp.parse(r.text.splitlines())
            crawl_delay = rp.crawl_delay(self.user_agent)
            if crawl_delay:
                self.delay = max(self.delay, float(crawl_delay))   # only ever slower, never faster
        return rp

    def _get(self, url: str) -> requests.Response:
        """The rate limit: wait until `delay` seconds have passed since the previous request ended, then GET."""
        wait = self._next_request_at - time.monotonic()      # monotonic: a clock that never jumps (e.g. DST)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.session.get(url, timeout=self.timeout)
        finally:
            # In `finally`, so even a failed request counts: errors can't make us hammer a site.
            self._next_request_at = time.monotonic() + self.delay
