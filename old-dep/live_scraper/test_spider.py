"""
test_spider.py
Run with: scrapy runspider test_spider.py
Fetches ONE real article URL per source and prints the parsed result.
Use this to verify selectors are working before running the full spider.
"""

import scrapy
from urllib.parse import urlparse


# One known article URL per source — update these if they go stale
TEST_URLS = {
    'protothema.gr':  'https://www.protothema.gr/politics/article/1471234/',
    'iefimerida.gr':  'https://www.iefimerida.gr/politiki/mitsotakis-interview',
    'kathimerini.gr': 'https://www.kathimerini.gr/world/564197230/sto-proskinio-politiki-via-kai-asfaleia/',
    'tanea.gr':       'https://www.tanea.gr/print/2026/04/28/opinions/xoris-arxigo-lfsta-plei-of/',
    'tovima.gr':      'https://www.tovima.gr/2026/04/28/society/efka-nea-epithesi-tou-idiou-drasti-sto-efeteio-anoikse-pyr-treis-traymaties/',
    'naftemporiki.gr':'https://www.naftemporiki.gr/society/2103585/loykareos-neoi-pyrovolismoi-apo-ton-drasti-tis-epithesis-ston-efka-kerameikoy-treis-traymaties/',
    'efsyn.gr':       'https://www.efsyn.gr/ellada/astynomiko/510182_synagermos-stin-athina-epithesi-89hronoy-me-karampina-se-efka-kerameikoy',
    'ethnos.gr':      'https://www.ethnos.gr/greece/article/406289/thrilerstokentrothsathhnasdyoepitheseismekarampinaseefkakerameikoykaiefeteioapo89xronorakosyllekth',
    'in.gr':          'https://www.in.gr/2026/04/28/go-fun/fizz/i-kori-tis-antzelina-tzoli-zaxara-leei-pos-einai-timi-tis-na-tin-apokalei-mitera-tis/',
    'skai.gr':        'https://www.skai.gr/news/politics/proso-olotaxos-gia-tin-maria-karystianou',
}

# Better: use the sitemap to get a real fresh URL automatically
SITEMAP_TEST_URLS = [
    'https://www.protothema.gr/sitemap/newsarticles/sitemap_index.xml',
    'https://www.iefimerida.gr/sitemap.xml',
    'https://www.kathimerini.gr/sitemap.xml',
    'https://www.tanea.gr/sitemap_index.xml',
    'https://www.tovima.gr/sitemap_index.xml',
    'https://www.naftemporiki.gr/sitemap.xml',
    'https://www.efsyn.gr/sitemap.xml',
    'https://www.ethnos.gr/sitemap.xml',
    'https://www.in.gr/sitemap.xml',
    'https://www.skai.gr/sitemap.xml',
]


class SpiderTester(scrapy.Spider):
    """
    Fetches the first sitemap of each source, grabs the first article URL,
    fetches it, parses it, and prints a formatted summary.
    Run: scrapy runspider test_spider.py
    """
    name = "spider_tester"

    custom_settings = {
        'DOWNLOAD_DELAY': 2.0,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'USER_AGENT': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        'LOG_LEVEL': 'WARNING',   # suppress scrapy noise, show only our prints
        'AUTOTHROTTLE_ENABLED': True,
    }

    def start_requests(self):
        for url in TEST_URLS.values():
            yield scrapy.Request(url, callback=self.parse_article)

    def parse_sitemap(self, response):
        # Find the first article URL in the sitemap
        # Works for both sitemap index and direct sitemap files
        locs = response.xpath('//*[local-name()="loc"]/text()').getall()

        article_url = None
        for loc in locs:
            loc = loc.strip()
            # Skip sub-sitemap files, find actual article URLs
            if 'sitemap' not in loc.lower() and loc.startswith('http'):
                article_url = loc
                break

        # If all locs are sub-sitemaps, follow the first one
        if not article_url:
            for loc in locs:
                loc = loc.strip()
                if loc.startswith('http'):
                    yield scrapy.Request(loc, callback=self.parse_sitemap)
                    return

        if article_url:
            yield scrapy.Request(article_url, callback=self.parse_article)

    def parse_article(self, response):
        domain = urlparse(response.url).netloc
        article = None

        if 'protothema.gr' in domain:
            article = self._extract_protothema(response)
        elif 'iefimerida.gr' in domain:
            article = self._extract_iefimerida(response)
        elif 'kathimerini.gr' in domain:
            article = self._extract_kathimerini(response)
        elif 'tanea.gr' in domain:
            article = self._extract_tanea(response)
        elif 'tovima.gr' in domain:
            article = self._extract_tovima(response)
        elif 'naftemporiki.gr' in domain:
            article = self._extract_naftemporiki(response)
        elif 'efsyn.gr' in domain:
            article = self._extract_efsyn(response)
        elif 'ethnos.gr' in domain:
            article = self._extract_ethnos(response)
        elif 'in.gr' in domain:
            article = self._extract_ingr(response)
        elif 'skai.gr' in domain:
            article = self._extract_skai(response)

        if article:
            self._print_article(article)
        else:
            print(f"\n⚠  No extractor matched for domain: {domain} | URL: {response.url}")

    def _print_article(self, article):
        source  = article.get('source', 'MISSING')
        title   = article.get('title') or 'MISSING'
        date    = article.get('date') or 'MISSING'
        url     = article.get('url', 'MISSING')
        text    = article.get('text') or ''

        ok_title = bool(article.get('title'))
        ok_date  = bool(article.get('date'))
        ok_text  = len(text) > 150

        status = "✅" if (ok_title and ok_date and ok_text) else "❌"

        print("\n" + "=" * 70)
        print(f"{status}  SOURCE  : {source}")
        print(f"   URL     : {url}")
        print(f"   TITLE   : {title[:80]}")
        print(f"   DATE    : {date}")
        print(f"   TEXT    : {text[:200]}..." if len(text) > 200 else f"   TEXT    : {text or 'EMPTY'}")
        print(f"   CHECKS  : title={'OK' if ok_title else 'MISSING'} | "
              f"date={'OK' if ok_date else 'MISSING'} | "
              f"text={'OK (' + str(len(text)) + ' chars)' if ok_text else 'TOO SHORT (' + str(len(text)) + ' chars)'}")
        print("=" * 70)

    def clean_text(self, text):
        if text:
            return ' '.join(text.split())
        return None

    # ── extractors (mirrors live_news_spider.py exactly) ─────────────────

    def _extract_protothema(self, response):
        return {
            'source': 'protothema.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': response.css('time::attr(datetime)').get(default='').strip(),
            'text': self.clean_text(' '.join(response.css('.cnt *::text').getall())),
        }

    def _extract_iefimerida(self, response):
        return {
            'source': 'iefimerida.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': response.css('.created::text, time::attr(datetime)').get(default='').strip(),
            'text': self.clean_text(' '.join(response.css(
                '.field--name-body p::text, .article-main-body p::text').getall())),
        }

    def _extract_kathimerini(self, response):
        return {
            'source': 'kathimerini.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': (response.css('time::attr(datetime)').get()
                     or response.css('meta[property="article:published_time"]::attr(content)').get()
                     or '').strip(),
            'text': self.clean_text(' '.join(response.css('.entry-content p::text').getall())),
        }

    def _extract_tanea(self, response):
        return {
            'source': 'tanea.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': (response.css('meta[property="article:published_time"]::attr(content)').get()
                     or response.css('meta[property="article:modified_time"]::attr(content)').get()
                     or response.css('time::attr(datetime)').get()
                     or '').strip(),
            'text': self.clean_text(' '.join(response.css('.main-content p::text').getall())),
        }

    def _extract_tovima(self, response):
        return {
            'source': 'tovima.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': (response.css('time::attr(datetime)').get()
                     or response.css('meta[property="article:published_time"]::attr(content)').get()
                     or '').strip(),
            'text': self.clean_text(' '.join(response.css('.main-content p::text').getall())),
        }

    def _extract_naftemporiki(self, response):
        return {
            'source': 'naftemporiki.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': (response.css('time::attr(datetime)').get()
                     or response.css('meta[property="article:published_time"]::attr(content)').get()
                     or response.css('.article-date::text').get()
                     or '').strip(),
            'text': self.clean_text(' '.join(response.css('.post-content p::text').getall())),
        }

    def _extract_efsyn(self, response):
        return {
            'source': 'efsyn.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': (response.css('time::attr(datetime)').get()
                     or response.css('meta[property="article:published_time"]::attr(content)').get()
                     or response.css('.field--name-created::text').get()
                     or '').strip(),
            'text': self.clean_text(' '.join(response.css('.article__body p::text').getall())),
        }

    def _extract_ethnos(self, response):
        return {
            'source': 'ethnos.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': (response.css('meta[property="article:modified_time"]::attr(content)').get()
                     or response.css('meta[property="article:published_time"]::attr(content)').get()
                     or response.css('time::attr(datetime)').get()
                     or '').strip(),
            'text': self.clean_text(' '.join(response.css('.article-content-container p::text').getall())),
        }

    def _extract_ingr(self, response):
        return {
            'source': 'in.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': (response.css('time::attr(datetime)').get()
                     or response.css('meta[property="article:published_time"]::attr(content)').get()
                     or response.css('.article-date time::attr(datetime)').get()
                     or '').strip(),
            'text': self.clean_text(' '.join(response.css('.main-content p::text').getall())),
        }

    def _extract_skai(self, response):
        return {
            'source': 'skai.gr', 'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': (response.css('time::attr(datetime)').get()
                     or response.css('meta[property="article:published_time"]::attr(content)').get()
                     or '').strip(),
            'text': self.clean_text(' '.join(response.css('.post-content p::text').getall())),
        }