import json
import argparse
import sys
from statistics import mean

def load_data(path):
    """Loads data from either a JSON (list) or JSONL file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                return json.loads(content)
            else:
                # Assume JSONL
                return [json.loads(line) for line in content.split('\n') if line.strip()]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

def extract_biases(data):
    """Maps URL to bias score, handling potential nesting or stringified JSON."""
    biases = {}
    for entry in data:
        # Use URL as primary key, fallback to title
        key = entry.get('url') or entry.get('title')
        if not key:
            continue
        
        labels = entry.get('ai_labels', {})
        # Handle case where ai_labels might be a stringified JSON (from legacy formats)
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except:
                continue
        
        bias = labels.get('bias')
        if bias is not None:
            try:
                biases[key] = float(bias)
            except ValueError:
                continue
    return biases

def calculate_mae(biases1, biases2):
    """Calculates Mean Absolute Error between two bias maps."""
    common_keys = set(biases1.keys()) & set(biases2.keys())
    
    if not common_keys:
        return None, 0
    
    errors = []
    for key in common_keys:
        err = abs(biases1[key] - biases2[key])
        errors.append(err)
    
    return mean(errors), len(common_keys)

def main():
    file1 = "dataset/reasoning_training/gemini_labels.jsonl"
    file2 = "dataset/reasoning_training/claude_labels.jsonl"

    print(f"Loading Source 1: {file1}...")
    data1 = load_data(file1)
    print(f"Loading Source 2: {file2}...")
    data2 = load_data(file2)

    biases1 = extract_biases(data1)
    biases2 = extract_biases(data2)

    mae, count = calculate_mae(biases1, biases2)

    print("\n" + "="*40)
    print("🏆 GROUND TRUTH VALIDATION DASHBOARD 🏆")
    print("="*40)
    print(f"Source A: {file1} ({len(biases1)} samples)")
    print(f"Source B: {file2} ({len(biases2)} samples)")
    print("-" * 40)

if __name__ == "__main__":
    main()
