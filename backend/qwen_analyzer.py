import sys
import json
import asyncio
import warnings
import re
from crawl4ai import AsyncWebCrawler
import vllm_client
warnings.filterwarnings("ignore")

# ==========================================
# --- YOUR CLEANING LOGIC ---
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
# --- YOUR SCRAPER ---
# ==========================================
async def scrape_with_crawl4ai(url):
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url, magic=True, exclude_external_links=True, exclude_social_media_links=True, word_count_threshold=15)
            raw_markdown = getattr(result, 'fit_markdown', result.markdown)
            
            if not raw_markdown: return None, "Crawl4AI connected, but found no text."
            clean_text = purify_markdown(raw_markdown)
            return clean_text[:4000], None 
            
    except Exception as e:
        return None, f"Crawl4AI failed: {str(e)}"

# ==========================================
# --- THE REACT BRIDGE ---
# ==========================================
async def analyze_article(url):
    # Step A: Run your scraper
    scraped_text, error = await scrape_with_crawl4ai(url)
    
    if error or not scraped_text:
        return {"title": "Scrape Failed", "polLean": "Center", "polScore": 50, "reasoning": error, "tags": []}

    # Step B: Trigger your partner's code to do the AI heavy lifting!
    raw_response = await asyncio.to_thread(vllm_client.call_vllm, scraped_text)
    ai_data = vllm_client.safe_parse(raw_response)
    
    # Step C: Format the result specifically for your React Dashboard
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
        "source": "vLLM Engine",
        "url": url
    }

# ==========================================
if __name__ == "__main__":
    # Force Python to use UTF-8 for Greek Characters
    sys.stdout.reconfigure(encoding='utf-8')
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No URL provided"}))
        sys.exit(1)
        
    target_url = sys.argv[1]
    final_data = asyncio.run(analyze_article(target_url))
    
    # Print final JSON for Node.js to catch
    print(json.dumps(final_data))