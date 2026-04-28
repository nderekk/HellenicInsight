import os
import sys
import io
import json
import asyncio
import warnings
import re
from openai import AsyncOpenAI
from crawl4ai import AsyncWebCrawler

# ==========================================
# --- SYSTEM SETUP ---
# ==========================================
# Force Python to output UTF-8 for Greek characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

# Initialize the AI Client (Works for Qwen via Ollama, vLLM, or Cloud APIs)
client = AsyncOpenAI(
    api_key=os.environ.get("QWEN_API_KEY", "your-api-key-here"), 
    base_url=os.environ.get("QWEN_BASE_URL", "http://localhost:11434/v1") # <-- CHANGE THIS TO YOUR QWEN URL
)

# ==========================================
# --- THE SMART DATA PURIFIER ---
# ==========================================
def purify_markdown(raw_text):
    if not raw_text:
        return ""

    # 1. The Smart Slice: Find the main headline (starts with "# ")
    start_index = raw_text.find('# ')
    if start_index != -1:
        raw_text = raw_text[start_index:] 
        
    # 2. Annihilate Markdown Images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', raw_text)
    
    # 3. Strip URLs but keep the text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    
    cleaned_lines = []
    
    # 4. Read the article line by line
    for line in text.split('\n'):
        line = line.strip()
        
        # The Footer Kill-Switch
        lower_line = line.lower()
        if "προσθηκη σχολιου" in lower_line or "διαβάστε ακόμα" in lower_line or "δημοφιλη" in lower_line or "read more" in lower_line:
            break 

        # Keep lines that have 5 or more words
        if len(line.split()) >= 5: 
            cleaned_lines.append(line)
            
    return "\n\n".join(cleaned_lines)

# ==========================================
# --- THE CRAWL4AI EXTRACTOR ---
# ==========================================
async def scrape_with_crawl4ai(url):
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(
                url=url,
                magic=True,                  
                exclude_external_links=True, 
                exclude_social_media_links=True, 
                word_count_threshold=15      
            )
            
            raw_markdown = getattr(result, 'fit_markdown', result.markdown)
            
            if not raw_markdown:
                return None, "Crawl4AI connected, but found no text."
                
            clean_text = purify_markdown(raw_markdown)
            
            # Save clean text to debug file
            with open("debug_scraped_article.md", "w", encoding="utf-8") as file:
                file.write(clean_text)
                
            return clean_text[:4000], None # Feed up to 4000 chars to Qwen
            
    except Exception as e:
        return None, f"Crawl4AI failed: {str(e)}"

# ==========================================
# --- THE QWEN AI ORCHESTRATOR ---
# ==========================================
async def analyze_article(url):
    scraped_text, error = await scrape_with_crawl4ai(url)
    
    if error or not scraped_text:
        return {
            "title": "Scrape Failed",
            "source": "Error Log",
            "polLean": "Center",
            "polScore": 50,
            "reasoning": f"Could not extract text. Error: {error}",
            "tags": []
        }

    system_prompt = """
    You are an expert Political Data Scientist and Computational Linguist specializing in Greek digital media and political discourse. Your task is to perform a deep-structure ideological analysis of Greek news articles.

    TASK:
    1. Analyze the provided Greek news article for political bias, framing, and ideological stance.
    2. Provide a concise reasoning in Greek (2-4 sentences) justifying the analysis.
    3. Extract 1-3 primary political entities (politicians, parties, institutions) targeted or discussed in the text.
    4. Assign a precise ideological leaning score on a continuous scale from 0.0 to 1.0.

    IDEOLOGICAL ANCHORS (Left vs Right & Populism vs Institutionalism):
    - 0.00 - 0.15: Far-Left (Radical systemic critique, anti-capitalist, anti-establishment/populist framing)
    - 0.16 - 0.35: Left (Socialist/Progressive focus, labor rights, strong state intervention)
    - 0.36 - 0.45: Center-Left (Social democratic leaning, moderate reformism, pro-EU)
    - 0.46 - 0.55: Center / Neutral (Strictly objective reporting, institutionalist, multi-perspective balance)
    - 0.56 - 0.65: Center-Right (Liberal-conservative, market-oriented, institutionalist/pro-EU)
    - 0.66 - 0.85: Right (Conservative, national focus, law and order, pro-business)
    - 0.86 - 1.00: Far-Right (Ultra-nationalist, nativist framing, reactionary/anti-systemic rhetoric)

    REASONING GUIDELINES (Greek):
    Your reasoning must identify:
    - Lexical choices (e.g., use of "λαϊκισμός", "δικαιωματισμός", "καθεστώς", "ελίτ").
    - Framing of political actors (who is portrayed as the protagonist/antagonist?).
    - Source selection (whose views are prioritized or omitted?).

    STRICT OUTPUT FORMAT:
    Return ONLY a valid JSON object. Do not include markdown code blocks, headers, or any text before/after the JSON. 

    JSON SCHEMA:
    {
      "title": "string (A short, accurate headline in Greek extracted from the text)",
      "reasoning": "string (in Greek, 2-4 sentences)",
      "primary_entities": ["string", "string"],
      "bias": float (0.00 to 1.00)
    }
    """

    try:
        response = await client.chat.completions.create(
            model="qwen2.5", # Ensure this matches your teammate's model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this article:\n\n{scraped_text}"}
            ],
            response_format={ "type": "json_object" }, 
            temperature=0.2 
        )
        
        # Parse the LLM's raw JSON response
        ai_raw_data = json.loads(response.choices[0].message.content)
        
        # Convert 0.0-1.0 float to React's 0-100 integer
        bias_float = ai_raw_data.get("bias", 0.5)
        pol_score_int = int(bias_float * 100)
        
        # Determine the React badge label
        if bias_float <= 0.35:
            pol_lean = "Left"
        elif bias_float >= 0.66:
            pol_lean = "Right"
        else:
            pol_lean = "Center"

        # Construct the final dictionary perfectly matched to your React UI
        final_react_data = {
            "title": ai_raw_data.get("title", f"Analysis of {url[:25]}..."),
            "polLean": pol_lean,
            "polScore": pol_score_int,
            "reasoning": ai_raw_data.get("reasoning", "Αποτυχία παραγωγής αιτιολόγησης."),
            "tags": ai_raw_data.get("primary_entities", []),
            "source": "Qwen Multi-Agent Pipeline",
            "url": url
        }
        
        return final_react_data

    except Exception as e:
        return {
            "title": "AI Inference Failed",
            "source": "System Error",
            "polLean": "Center",
            "polScore": 50,
            "reasoning": f"Text was scraped successfully, but the Qwen LLM crashed: {str(e)}",
            "tags": []
        }

# ==========================================
# --- MAIN EXECUTION ---
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No URL provided"}))
        sys.exit(1)
        
    target_url = sys.argv[1]
    final_data = asyncio.run(analyze_article(target_url))
    
    print(json.dumps(final_data))