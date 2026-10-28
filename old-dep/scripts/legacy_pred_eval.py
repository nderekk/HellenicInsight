import json
from statistics import mean

with open("datasets/preds/evaluation_resultsv0.1.json", 'r') as f:
    file = json.load(f)
  
scores = {  
    "sentiment": {
        'negative': -1,
        'neutral': 0,
        'positive': 1
    },
    "ideological_leaning": {
        "far-left": -3,
        "left": -2,
        "center-left":-1,
        "center":0,
        "center-right":1,
        "right":2,
        "far-right":3,
        "neutral":0
    },
    "establishment_stance": {
        "pro-government":2,
        "anti-government":-2,
        "systemic":1,
        "anti-systemic":-1,
        "neutral":0,
    }
}
    
total_errors = []
fallacy_matches = 0
valid_predictions = 0

for count, entry in enumerate(file):
    try:
        labels = json.loads(entry['actual_ground_truth']) if isinstance(entry['actual_ground_truth'], str) else entry['actual_ground_truth']
        preds = json.loads(entry['model_prediction']) if isinstance(entry['model_prediction'], str) else entry['model_prediction']

        print(labels)
        print(preds)
        
        temp_err = 0
        
        for key in labels.keys():
            pred_val = preds[key]
            label_val = labels[key]
            
            if key == 'primary_fallacy':
                is_wrong = (pred_val != label_val)
                temp_err += is_wrong
                if not is_wrong:
                    fallacy_matches += 1
                # print(f'temp error {count}: ', is_wrong)
                temp_err += is_wrong
            else: 
                pred_num = scores[key].get(pred_val, 5) 
                label_num = scores[key][label_val]
                distance = abs(pred_num - label_num)
                print(f'temp error {count}: ', distance)
                temp_err += distance
                
        total_errors.append(temp_err)
        valid_predictions += 1
    except Exception as e:
        print(f"Skipping article {count} due to malformed JSON or error: {e}")
        # Add a massive penalty for crashing
        total_errors.append(10)
    
mean_error = mean(total_errors) if total_errors else 0
fallacy_accuracy = (fallacy_matches / valid_predictions) * 100 if valid_predictions else 0

print("=======================================")
print("🏆 PIPELINE EVALUATION DASHBOARD 🏆")
print("=======================================")
print(f"Articles Evaluated: {valid_predictions} / {len(file)}")
print(f"Mean Absolute Error (Distance): {mean_error:.2f} points per article")
print(f"Exact Fallacy Match Accuracy: {fallacy_accuracy:.1f}%")
print("=======================================")

if mean_error < 1.5:
    print("STATUS: OUTSTANDING. Model is highly calibrated to ground truth.")
elif mean_error < 3.0:
    print("STATUS: GOOD. Model understands the scale but misses nuance.")
else:
    print("STATUS: NEEDS WORK. High deviation or frequent JSON hallucinations.")