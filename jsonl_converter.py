import json

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