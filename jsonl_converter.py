import json

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

# 1. Load your API-labeled dataset
with open('datasets/labeled_dataset.json', 'r', encoding='utf-8') as f:
    labeled_data = json.load(f)

# 2. Open a new file in 'write' mode with the .jsonl extension
with open('datasets/training_data.jsonl', 'w', encoding='utf-8') as outfile:
    for item in labeled_data:
        # We only want to train on items that actually succeeded and have labels
        if "ai_labels" in item:
            
            # Reconstruct the exact prompt the AI saw
            user_text = f"ΤΙΤΛΟΣ: {item['title']}\nΚΕΙΜΕΝΟ: {item['text']}"
            
            # Convert the JSON labels BACK into a string, because the model 
            # is learning to generate text that *looks* like a JSON object.
            assistant_text = json.dumps(item["ai_labels"], ensure_ascii=False)
            
            
            
            # The standard Instruction-Tuning format
            training_row = {
                "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
                ]
            }
            
            # Write it as a single line, followed by a newline character
            json.dump(training_row, outfile, ensure_ascii=False)
            outfile.write('\n')

print("Conversion complete! training_data.jsonl is ready for the GPU.")