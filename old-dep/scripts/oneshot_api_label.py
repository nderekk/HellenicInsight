import vertexai
from vertexai.generative_models import GenerativeModel
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Vertex AI
if os.getenv("project_key"):
    vertexai.init(project=os.getenv("project_key"), location="europe-west4")
else:
    print("Warning: 'project_key' not found in environment variables.")

# System Prompt from api_label.py
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

def label_oneshot(text, title="User Article"):
    model = GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=[system_prompt]
    )
    
    article_content = f"ΤΙΤΛΟΣ: {title}\nΚΕΙΜΕΝΟ: {text}"
    
    try:
        response = model.generate_content(
            article_content,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

# Example usage for hardcoded articles:
# articles = [
#     {"title": "", "text": "Ο κουλης ειναι τρελος. θα μας καταστρεψει ολους!!!"},
# ]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If text is provided as an argument
        input_text = " ".join(sys.argv[1:])
        label_oneshot(input_text)
    else:
        # Otherwise, ask for input
        print("Please enter the article text (Press Ctrl+D or Ctrl+Z to finish):")
        input_text = sys.stdin.read()
        if input_text.strip():
            label_oneshot(input_text)
        else:
            print("No text provided.")
