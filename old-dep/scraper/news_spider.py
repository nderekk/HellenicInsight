from scrapy.spiders import SitemapSpider
from urllib.parse import urlparse
import re

class HistoricalGreekNewsSpider(SitemapSpider):
    name = "historical_greek_news"
    
    custom_settings = {
        'CLOSESPIDER_ITEMCOUNT': 8000, 
        'DOWNLOAD_DELAY': 1.5, 
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2, 
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        },
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,
        'FEED_EXPORT_ENCODING': 'utf-8'
    }

    sitemap_urls = [
        'https://www.iefimerida.gr/sitemap.xml',
        'file:///Users/koules/Developer/side_projects/keifer_hackathon/scraper/protothema_sitemap.xml'
    ]
    
    # NEW: The Topic Filter!
    # The spider will only parse URLs that contain these specific category paths.
    sitemap_rules = [
        (r'/politics/|/economy/|/world/|/greece/', 'parse'),     # Protothema URL structures
        (r'/politiki/|/oikonomia/|/kosmos/|/ellada/', 'parse')    # Iefimerida URL structures
    ]

    def parse(self, response):
        domain = urlparse(response.url).netloc
        article_data = None

        if 'protothema.gr' in domain:
            article_data = self.extract_protothema(response)
        elif 'iefimerida.gr' in domain:
            article_data = self.extract_iefimerida(response)

        if article_data and article_data.get('date'):
            article_year = self.extract_year(article_data['date'])
            
            if article_year in [2023, 2024, 2025, 2026]:  # Adjust the years as needed
                yield article_data

    def extract_protothema(self, response):
        return {
            'source': 'protothema.gr',
            'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': response.css('time::attr(datetime)').get(default='').strip(),
            # Grabbing all text, we will split out the cookie banner in Pandas
            'text': self.clean_text(' '.join(response.css('.cnt *::text').getall()))
        }

    def extract_iefimerida(self, response):
        return {
            'source': 'iefimerida.gr',
            'url': response.url,
            'title': self.clean_text(response.css('h1::text').get()),
            'date': response.css('.created::text, time::attr(datetime)').get(default='').strip(),
            'text': self.clean_text(' '.join(response.css('.field--name-body p::text, .article-main-body p::text').getall()))
        }

    def extract_year(self, date_string):
        match = re.search(r'\b(20\d{2})\b', date_string)
        return int(match.group(1)) if match else None

    def clean_text(self, text):
        if text:
            return ' '.join(text.split())
        return None