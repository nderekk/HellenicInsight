import vertexai
from vertexai.generative_models import GenerativeModel
import json
import time
import os
import anthropic
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Initialize the Enterprise Connection
# The SDK automatically finds the credentials you just created in the terminal!
if os.getenv("project_key"):
    vertexai.init(project=os.getenv("project_key"), location="europe-west4")

# 2. Your Master Prompt
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
  "reasoning": "string (in Greek, 2-4 sentences)",
  "primary_entities": ["string", "string"],
  "bias": float (0.00 to 1.00)
}

EXAMPLE OUTPUT:
{"reasoning": "Το άρθρο χρησιμοποιεί έντονα φορτισμένους όρους όπως 'νεοφιλελεύθερη λαίλαπα' και εστιάζει αποκλειστικά σε ανακοινώσεις συνδικάτων χωρίς να παραθέτει την κυβερνητική θέση, γεγονός που υποδηλώνει σαφή αριστερή/αντισυστημική απόκλιση.", "primary_entities": ["Κυβέρνηση", "ΓΣΕΕ"], "bias": 0.18}
"""

def save_to_jsonl(item, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_processed_titles(output_file):
    processed_titles = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed_titles.add(item.get("title"))
                except json.JSONDecodeError:
                    continue
    return processed_titles

def label_gemini(test_articles, output_file='datasets/labeled_dataset.jsonl'):
    print(f"--- Starting Gemini Labeling -> {output_file} ---")
    model = GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=[system_prompt]
    )
    
    processed_titles = get_processed_titles(output_file)
    articles_to_process = [item for item in test_articles if item.get("title") not in processed_titles]
    print(f"Processing {len(articles_to_process)} new articles.")

    for i, item in enumerate(articles_to_process, 1):
        title = item.get("title", "")
        text = item.get("text", "")
        article_content = f"ΤΙΤΛΟΣ: {title}\nΚΕΙΜΕΝΟ: {text}"
        
        success = False
        while not success:
            try:
                response = model.generate_content(
                    article_content,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1
                    }
                )
                parsed_json = json.loads(response.text) 
                item_copy = item.copy()
                item_copy["ai_labels"] = parsed_json
                save_to_jsonl(item_copy, output_file)
                print(f"Gemini Success: {title[:30]}...")
                success = True
                time.sleep(1)
            except Exception as e:
                print(f"Gemini Error: {e}")
                time.sleep(5)

def label_claude(test_articles, output_file='datasets/training/claude_labels.jsonl'):
    print(f"--- Starting Claude Labeling -> {output_file} ---")
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    processed_titles = get_processed_titles(output_file)
    articles_to_process = [item for item in test_articles if item.get("title") not in processed_titles]
    print(f"Processing {len(articles_to_process)} new articles.")

    for i, item in enumerate(articles_to_process, 1):
        title = item.get("title", "")
        text = item.get("text", "")
        
        success = False
        while not success:
            try:
                message = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": f"ΤΙΤΛΟΣ: {title}\nΚΕΙΜΕΝΟ: {text}"}
                    ]
                )
                # Anthropic doesn't have native JSON mode in all models via simple flag yet, 
                # but we asked for it in system prompt.
                content = message.content[0].text
                parsed_json = json.loads(content)
                item_copy = item.copy()
                item_copy["ai_labels"] = parsed_json
                save_to_jsonl(item_copy, output_file)
                print(f"Claude Success: {title[:30]}...")
                success = True
                time.sleep(1)
            except Exception as e:
                print(f"Claude Error: {e}")
                time.sleep(5)

def label_chatgpt(test_articles, output_file='datasets/training/gpt_labels.jsonl'):
    print(f"--- Starting ChatGPT Labeling -> {output_file} ---")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    processed_titles = get_processed_titles(output_file)
    articles_to_process = [item for item in test_articles if item.get("title") not in processed_titles]
    print(f"Processing {len(articles_to_process)} new articles.")

    for i, item in enumerate(articles_to_process, 1):
        title = item.get("title", "")
        text = item.get("text", "")
        
        success = False
        while not success:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"ΤΙΤΛΟΣ: {title}\nΚΕΙΜΕΝΟ: {text}"}
                    ],
                    response_format={ "type": "json_object" },
                    temperature=0.1
                )
                parsed_json = json.loads(response.choices[0].message.content)
                item_copy = item.copy()
                item_copy["pred"] = parsed_json
                save_to_jsonl(item_copy, output_file)
                print(f"ChatGPT Success: {title[:30]}...")
                success = True
                time.sleep(1)
            except Exception as e:
                print(f"ChatGPT Error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    with open('datasets/final_unlabeled_dataset.json', 'r', encoding='utf-8') as f:
        unlabeled_dataset = json.load(f)
    
    # We take the first 20 for this validation task
    # validation_subset = unlabeled_dataset[:3000]
    
    # Run Gemini for the subset as well (into a validation file)
    label_gemini(validation_subset, output_file='datasets/training/200sample.jsonl')
    
    # Run Claude
    # label_claude(validation_subset, output_file='datasets/training/500sample.jsonl')
    
    # Run ChatGPT
    # label_chatgpt(validation_subset)
