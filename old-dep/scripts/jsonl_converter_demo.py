import json

# 1. Load your API-labeled dataset
with open('datasets/final_unlabeled_dataset.json', 'r', encoding='utf-8') as f:
    labeled_data = json.load(f)
    

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

# 2. Open a new file in 'write' mode with the .jsonl extension
with open('datasets/jsonl_demo.jsonl', 'w', encoding='utf-8') as outfile:
    for item in labeled_data:
        # We only want to train on items that actually succeeded and have labels
        if "source" in item:
            
            # Reconstruct the exact prompt the AI saw
            user_text = f"ΤΙΤΛΟΣ: {item['title']}\nΚΕΙΜΕΝΟ: {item['text']}"
            
            # Convert the JSON labels BACK into a string, because the model 
            # is learning to generate text that *looks* like a JSON object.            
            
            
            # The standard Instruction-Tuning format
            training_row = {
                **item
            }
            
            # Write it as a single line, followed by a newline character
            json.dump(training_row, outfile, ensure_ascii=False, default=set_default)
            outfile.write('\n')

print("Conversion complete! training_data.jsonl is ready for the GPU.")