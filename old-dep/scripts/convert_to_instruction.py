import json
import os

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

def convert_jsonl_to_instruction(input_path, output_path):
    print(f"Reading from {input_path}...")
    count = 0
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # We only want to train on items that have labels
            if "ai_labels" in item and item["ai_labels"]:
                # Reconstruct the exact prompt the AI saw
                user_text = f"ΤΙΤΛΟΣ: {item.get('title', 'N/A')}\nΚΕΙΜΕΝΟ: {item.get('text', 'N/A')}"
                
                # Convert the JSON labels BACK into a string
                assistant_text = json.dumps(item["ai_labels"], ensure_ascii=False)
                
                # The standard Instruction-Tuning format (ChatML style)
                training_row = {
                    "messages": [
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ]
                }
                
                row_str = json.dumps(training_row, ensure_ascii=False)
                outfile.write(row_str + '\n')
                count += 1

    print(f"Conversion complete! {count} examples written to {output_path}")

if __name__ == "__main__":
    input_file = 'datasets/training/2sample.jsonl'
    output_file = 'datasets/training/2instruction_training_data.jsonl'
    
    if os.path.exists(input_file):
        convert_jsonl_to_instruction(input_file, output_file)
    else:
        print(f"Error: {input_file} not found.")
