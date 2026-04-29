import os
import re
import json
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from scrapy.spiders import SitemapSpider
from scrapy import signals


class LiveGreekNewsSpider(SitemapSpider):
    name = "live_greek_news"

    custom_settings = {
        'CLOSESPIDER_ITEMCOUNT': 8000,
        'CLOSESPIDER_TIMEOUT': 600,        # hard stop after 10 minutes
        'DOWNLOAD_DELAY': 3.0,
        'RANDOMIZE_DOWNLOAD_DELAY': True,          # randomize between 1.5x–2x DOWNLOAD_DELAY
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'USER_AGENT': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7',
        },
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,
        'AUTOTHROTTLE_MAX_DELAY': 10.0,
        'RETRY_TIMES': 2,
        'HTTPCACHE_ENABLED': False,
        'FEED_EXPORT_ENCODING': 'utf-8',
        # Rotate through a few common user agents to reduce fingerprinting
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
        },
    }

    sitemap_urls = [
        'https://www.iefimerida.gr/sitemap.xml',
        'https://www.protothema.gr/sitemap/newsarticles/sitemap_index.xml',
        'https://www.kathimerini.gr/sitemap.xml',
        'https://www.tanea.gr/wp-content/uploads/json/sitemap-news.xml',
        'https://www.tovima.gr/sitemap_index.xml',
    ]

    sitemap_rules = [
        (r'protothema\.gr.*/politics/|protothema\.gr.*/economy/|protothema\.gr.*/world/|protothema\.gr.*/greece/', 'parse'),
        (r'iefimerida\.gr.*/politiki/|iefimerida\.gr.*/oikonomia/|iefimerida\.gr.*/kosmos/|iefimerida\.gr.*/ellada/', 'parse'),
        (r'kathimerini\.gr.*/\d+/', 'parse'),
        (r'tanea\.gr.*/\d{4}/\d{2}/\d{2}/', 'parse'),
        (r'tovima\.gr.*/\d{4}/\d{2}/\d{2}/', 'parse'),
    ]

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(LiveGreekNewsSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def __init__(self, *args, **kwargs):
        super(LiveGreekNewsSpider, self).__init__(*args, **kwargs)
        self.state_file = 'last_scraped_time.json'
        
        self.cutoffs = {}
        self.newest_timestamps = {}
        
        now = datetime.now(timezone.utc)
        self.default_cutoff = now - timedelta(hours=2)
        self.sitemap_counts = {}  # sub-sitemap counter per domain (no-lastmod flood guard)
        max_lookback = now - timedelta(hours=2)  # never scrape more than 2 hours back

        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                try:
                    saved_times = json.load(f)
                    for dom, ts in saved_times.items():
                        dt = datetime.fromisoformat(ts)
                        if not dt.tzinfo:
                            dt = dt.replace(tzinfo=timezone.utc)
                        self.cutoffs[dom] = max(dt, max_lookback)
                except Exception:
                    pass

    def sitemap_filter(self, entries):
        for entry in entries:
            lastmod = entry.get('lastmod')
            loc     = entry.get('loc', '')
            loc_lower = loc.lower()
            domain  = urlparse(loc).netloc.replace('www.', '')

            # Skip non-article sub-sitemaps (author, category, tag pages)
            if any(s in loc_lower for s in ['author', 'categor', 'tag', 'page-sitemap', 'product']):
                continue

            cutoff = self.default_cutoff
            for dom, dt in self.cutoffs.items():
                if dom in domain:
                    cutoff = dt
                    break

            if lastmod:
                try:
                    clean_str  = lastmod.replace('Z', '+00:00')
                    entry_date = datetime.fromisoformat(clean_str)
                    if not entry_date.tzinfo:
                        entry_date = entry_date.replace(tzinfo=timezone.utc)
                    if entry_date >= cutoff:
                        yield entry
                except ValueError:
                    yield entry
            else:
                # No lastmod: skip URLs that contain a past year (/2024/, /2023/, …)
                year_match = re.search(r'/(\d{4})/', loc)
                if year_match and int(year_match.group(1)) < datetime.now(timezone.utc).year:
                    pass  # clearly historical
                elif loc_lower.endswith('.xml'):
                    # Sub-sitemap pointer without lastmod (e.g. protothema NewsArticles/N.xml)
                    # Cap at 3 per domain so we don't flood through hundreds of old files
                    count = self.sitemap_counts.get(domain, 0)
                    if count < 3:
                        self.sitemap_counts[domain] = count + 1
                        yield entry
                else:
                    yield entry  # regular article URL, pass through

    def spider_closed(self, spider):
        final_times = self.cutoffs.copy()
        for dom, dt in self.newest_timestamps.items():
            # Only update if the new timestamp is strictly newer
            if dom not in final_times or dt > final_times[dom]:
                final_times[dom] = dt
                
        if final_times:
            with open(self.state_file, 'w') as f:
                json.dump({k: v.isoformat() for k, v in final_times.items()}, f, indent=4)

    def parse(self, response):
        domain = urlparse(response.url).netloc
        article_data = None

        if 'protothema.gr' in domain:
            article_data = self.extract_protothema(response)
        elif 'iefimerida.gr' in domain:
            article_data = self.extract_iefimerida(response)
        elif 'kathimerini.gr' in domain:
            article_data = self.extract_kathimerini(response)
        elif 'tanea.gr' in domain:
            article_data = self.extract_tanea(response)
        elif 'tovima.gr' in domain:
            article_data = self.extract_tovima(response)

        if article_data and article_data.get('date'):
            article_date_obj = self.parse_datetime(article_data['date'])
            if article_date_obj:
                if not article_date_obj.tzinfo:
                    article_date_obj = article_date_obj.replace(tzinfo=timezone.utc)
                
                source_domain = article_data['source']
                cutoff = self.cutoffs.get(source_domain, self.default_cutoff)
                
                if article_date_obj > cutoff:
                    current_newest = self.newest_timestamps.get(source_domain)
                    if not current_newest or article_date_obj > current_newest:
                        self.newest_timestamps[source_domain] = article_date_obj
                    yield article_data

    # ── EXISTING EXTRACTORS ──────────────────────────────────────────────

    def extract_protothema(self, response):
        return {
            'source': 'protothema.gr',
            'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': response.css('time::attr(datetime)').get(default='').strip(),
            'text': self.clean_text(' '.join(response.css('.cnt *::text').getall())),
        }

    def extract_iefimerida(self, response):
        return {
            'source': 'iefimerida.gr',
            'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': response.css('.created::text, time::attr(datetime)').get(default='').strip(),
            'text': self.clean_text(
                ' '.join(response.css(
                    '.field--name-body p::text, .article-main-body p::text'
                ).getall())
            ),
        }

    # ── NEW EXTRACTORS ───────────────────────────────────────────────────

    def extract_kathimerini(self, response):
        return {
            'source': 'kathimerini.gr',
            'url': response.url,
            'title': self.clean_text(
                response.css('h1::text').get()
            ),
            'date': (
                response.css('time.entry-date::attr(datetime)').get()
                or response.css('meta[property="article:published_time"]::attr(content)').get()
                or ''
            ).strip(),
            'text': self.clean_text(
                ' '.join(response.css('.entry-content p::text').getall())
            ),
        }

    def extract_tanea(self, response):
        return {
            'source': 'tanea.gr',
            'url': response.url,
            'title': self.clean_text(
                response.css('h1::text').get()
            ),
            'date': (
                response.css('meta[property="article:published_time"]::attr(content)').get()
                or response.css('meta[property="article:modified_time"]::attr(content)').get()
                or response.css('time::attr(datetime)').get()
                or ''
            ).strip(),
            'text': self.clean_text(
                ' '.join(response.css('.main-content p::text').getall())
            ),
        }

    def extract_tovima(self, response):
        return {
            'source': 'tovima.gr',
            'url': response.url,
            'title': self.clean_text(
                response.css('h1::text').get()
            ),
            'date': (
                response.css('time::attr(datetime)').get()
                or response.css('meta[property="article:published_time"]::attr(content)').get()
                or ''
            ).strip(),
            'text': self.clean_text(
                ' '.join(response.css('.main-content p::text').getall())
            ),
        }

    # ── HELPERS ──────────────────────────────────────────────────────────

    def parse_datetime(self, date_string):
        try:
            clean_str = date_string.replace('Z', '+00:00')
            return datetime.fromisoformat(clean_str)
        except ValueError:
            match = re.search(r'\b(20\d{2})-(\d{2})-(\d{2})\b', date_string)
            if match:
                return datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    tzinfo=timezone.utc,
                )
            return None

    def clean_text(self, text):
        if text:
            return ' '.join(text.split())
        return None