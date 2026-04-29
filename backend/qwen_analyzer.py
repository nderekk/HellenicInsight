import sys
import json
import asyncio
import warnings
import re
from aiohttp import web
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
import vllm_client
warnings.filterwarnings("ignore")

# ==========================================
# --- CLEANING LOGIC ---
# ==========================================
def purify_markdown(raw_text):
    if not raw_text: return ""

    start_index = raw_text.find('# ')
    if start_index != -1: raw_text = raw_text[start_index:]

    text = re.sub(r'!\[.*?\]\(.*?\)', '', raw_text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'http[s]?://\S+', '', text)

    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in ["προσθηκη σχολιου", "διαβάστε ακόμα", "δημοφιλη", "read more"]):
            break
        if len(line.split()) >= 5:
            cleaned_lines.append(line)

    return "\n\n".join(cleaned_lines)

# ==========================================
# --- PERSISTENT CRAWLER (initialized once on startup) ---
# ==========================================
_crawler: AsyncWebCrawler | None = None

_browser_config = BrowserConfig(
    headless=True,
    text_mode=True,   # skip images/CSS/fonts
    verbose=False,
)

_run_config = CrawlerRunConfig(
    exclude_external_links=True,
    exclude_social_media_links=True,
    word_count_threshold=15,
    page_timeout=20000,
)

async def scrape_with_crawl4ai(url):
    try:
        result = await _crawler.arun(url=url, config=_run_config)
        raw_markdown = getattr(result, 'fit_markdown', result.markdown)
        if not raw_markdown:
            return None, "Crawl4AI connected, but found no text."
        return purify_markdown(raw_markdown)[:4000], None
    except Exception as e:
        return None, f"Crawl4AI failed: {str(e)}"

# ==========================================
# --- ANALYSIS LOGIC ---
# ==========================================
async def analyze_article(url):
    scraped_text, error = await scrape_with_crawl4ai(url)

    if error or not scraped_text:
        return {"title": "Scrape Failed", "polLean": "Center", "polScore": 50, "reasoning": error, "tags": []}

    raw_response = await asyncio.to_thread(vllm_client.call_vllm, scraped_text)
    ai_data = vllm_client.safe_parse(raw_response)

    bias_float = ai_data.get("bias", 0.5)
    if bias_float == -1.0: bias_float = 0.5

    pol_score_int = int(bias_float * 100)
    if bias_float <= 0.35: pol_lean = "Left"
    elif bias_float >= 0.66: pol_lean = "Right"
    else: pol_lean = "Center"

    return {
        "title": ai_data.get("title", f"Analysis of {url[:25]}..."),
        "polLean": pol_lean,
        "polScore": pol_score_int,
        "reasoning": ai_data.get("reasoning", "Error generating reasoning."),
        "tags": ai_data.get("primary_entities", []),
        "source": url[:25],
        "url": url
    }

# ==========================================
# --- AIOHTTP SERVER ---
# ==========================================
async def handle_analyze(request):
    try:
        body = await request.json()
        url = body.get("url")
    except Exception:
        return web.Response(status=400, text="Invalid JSON")

    if not url:
        return web.Response(status=400, text="url is required")

    result = await analyze_article(url)
    return web.Response(
        text=json.dumps(result, ensure_ascii=False),
        content_type='application/json'
    )

async def start_server():
    global _crawler
    print("[Analyzer] Starting browser...", flush=True)
    _crawler = AsyncWebCrawler(config=_browser_config)
    await _crawler.start()
    print("[Analyzer] Browser ready.", flush=True)

    app = web.Application()
    app.router.add_post('/analyze', handle_analyze)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 5002)
    await site.start()
    print("[Analyzer] Listening on http://127.0.0.1:5002", flush=True)

    await asyncio.Event().wait()  # run forever

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    asyncio.run(start_server())
