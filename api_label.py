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
You are an expert political data scientist analyzing Greek digital media. Read the provided Greek news article and extract the political ideological leaning of the text on a continuous numerical scale from 0.0 to 1.0.

DEFINITIONS & EXPLICIT ANCHORS:
- 0.00 to 0.15: Far-Left
- 0.16 to 0.35: Left
- 0.36 to 0.45: Center-Left
- 0.46 to 0.55: Center (or strictly Neutral/Objective reporting)
- 0.56 to 0.65: Center-Right
- 0.66 to 0.85: Right
- 0.86 to 1.00: Far-Right

INSTRUCTIONS:
Assign a precise decimal value based on the severity of the bias. For example, a moderately right-wing article might be 0.72, while an extreme left-wing article might be 0.05.

STRICT OUTPUT RULES:
You must respond ONLY with a valid, parsable JSON object. Do NOT include markdown formatting, code blocks, explanations, or any trailing text. 

EXAMPLE OUTPUT:
{"bias": 0.72}
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
    validation_subset = unlabeled_dataset[4000:4020]
    
    # Run Gemini for the subset as well (into a validation file)
    label_gemini(validation_subset, output_file='datasets/training/2sample.jsonl')
    
    # Run Claude
    # label_claude(validation_subset)
    
    # Run ChatGPT
    # label_chatgpt(validation_subset)
