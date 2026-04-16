import vertexai
from vertexai.generative_models import GenerativeModel
import json
import time

# 1. Initialize the Enterprise Connection
# The SDK automatically finds the credentials you just created in the terminal!
vertexai.init(project="REMOVED_SECRET", location="europe-west4")

# 2. Your Master Prompt (Insert your Taxonomy from Hour 1 here)
system_prompt = """
You are an expert political data scientist analyzing Greek digital media. Read the provided Greek news article and extract the sentiment, ideological leaning, establishment stance, and the primary logical fallacy present (if any).

DEFINITIONS & ALLOWED VALUES:
- "sentiment": ONLY use "positive", "negative", or "neutral". (Based on the author's tone toward the subject).
- "ideological_leaning": ONLY use "far-left", "left", "center-left", "center", "center-right", "right", "far-right", or "neutral".
- "establishment_stance": ONLY use "pro-government", "anti-government", "systemic", "anti-systemic", or "neutral".
- "primary_fallacy": Choose the ONE most dominant fallacy from this exact list. If none exist, output "none".
  ALLOWED FALLACIES: ["Επίθεση στο Πρόσωπο", "Επιχείρημα του 'Και για το άλλο;'", "Αχυράνθρωπος", "Ψευδές Δίλημμα", "Ολισθηρή Πλαγιά", "Επίκληση στον Φόβο", "Επιλεκτική Χρήση Στοιχείων", "Επίκληση στην Αυθεντία", "Σφάλμα Αιτιότητας", "Βιαστική Γενίκευση", "Επίκληση στην Πλειοψηφία", "Παγιδευμένη Ερώτηση", "Αντιπερισπασμός", "Κανένας αληθινός...", "Φαύλος Κύκλος", "Ανεκδοτολογική Μαρτυρία", "Επίκληση στην Παράδοση", "Ψευδής Αναλογία", "Επίκληση στην Άγνοια", "Φαινόμενο του Φωτοστέφανου", "none"]

STRICT OUTPUT RULES:
You must respond ONLY with a valid, parsable JSON object. Do NOT include markdown formatting, code blocks (like ```json), explanations, or any trailing text. 

EXAMPLE OUTPUT:
{"sentiment": "negative", "ideological_leaning": "right", "establishment_stance": "anti-government", "primary_fallacy": "Επίκληση στον Φόβο"}
"""

model = GenerativeModel(
    'gemini-2.5-flash',
    system_instruction=[system_prompt] # <--- It goes here in Vertex!
)

with open('datasets/final_unlabeled_dataset.json', 'r') as f:
    unlabeled_dataset = json.load(f)
    
test_articles = unlabeled_dataset[:50]

# 4. The Distillation Loop
labeled_dataset = []

for item in test_articles:
    title = item.get("title", "")
    text = item.get("text", "")
    article_content = f"ΤΙΤΛΟΣ: {title}\nΚΕΙΜΕΝΟ: {text}"
    
    success = False
    
    # Retry loop: Try up to 3 times for a single article
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
            item["ai_labels"] = parsed_json
            labeled_dataset.append(item)
            
            print(f"Success! Labeled: {title[:30]}...")
            success = True # Breaks the while loop
            print(f"AI Labels for '{title[:30]}': {parsed_json}")
            time.sleep(1)  # Increased base sleep to 5 seconds (12 RPM)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10) # Simple backoff

with open('datasets/labeled_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(labeled_dataset, f, ensure_ascii=False, indent=2)

print("Distillation complete. Dataset saved!")