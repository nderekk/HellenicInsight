import scrapy
import re

class EfsynArchiveSpider(scrapy.Spider):
    name = "efsyn_archive"
    
    # The same Polite & Stealth settings we used before
    custom_settings = {
        'CLOSESPIDER_ITEMCOUNT': 4000, # Grab roughly a third of your total dataset goal
        'DOWNLOAD_DELAY': 1.5,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        },
        'AUTOTHROTTLE_ENABLED': True,
        'FEED_EXPORT_ENCODING': 'utf-8'
    }

    def start_requests(self):
        """
        Instead of a sitemap, we generate our own map by looping through
        the deep pages of Efsyn's main categories.
        """
        categories = ['politiki', 'ellada', 'oikonomia', 'kosmos']
        
        for category in categories:
            # We start deep in the pagination. 
            # Page 400-800 usually correlates to articles from a few years ago.
            for page in range(1, 800):
                url = f'https://www.efsyn.gr/{category}?page={page}'
                yield scrapy.Request(url, callback=self.parse_listing)

    def parse_listing(self, response):
        """
        This finds all the article links on the category page and follows them.
        """
        # Efsyn wraps article teasers in <article> tags
        article_links = response.css('article a::attr(href)').getall()
        
        for link in set(article_links): # Use set() to remove duplicate links on the same page
            yield response.follow(link, callback=self.parse_article)

    def parse_article(self, response):
        """
        Our standard extraction logic with the date filter applied.
        """
        date_str = response.css('time::attr(datetime)').get(default='').strip()
        article_year = self.extract_year(date_str)

        # Only save it if it falls in our 3-year historical window
        if article_year in [2023, 2024, 2025, 2026]:
            title = response.css('h1::text').get()
            raw_text = ' '.join(response.css('.article__body p::text, .field--name-body p::text').getall())
            
            if title and raw_text:
                yield {
                    'source': 'efsyn.gr',
                    'url': response.url,
                    'title': ' '.join(title.split()),
                    'date': date_str,
                    'text': ' '.join(raw_text.split())
                }

    def extract_year(self, date_string):
        match = re.search(r'\b(20\d{2})\b', date_string)
        return int(match.group(1)) if match else None