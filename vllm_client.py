import json
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
VLLM_URL = f"http://{os.environ['VLLM_HOST']}:{os.environ['VLLM_PORT']}/v1/chat/completions"
API_KEY = os.environ["VLLM_API_KEY"]
MODEL = os.environ["VLLM_MODEL"]
MAX_WORKERS = 16       # parallel HTTP requests to vLLM
TIMEOUT = 120          # seconds per request
TEMPERATURE = 0.1
MAX_TOKENS = 512

SYSTEM_PROMPT = """You are an expert Political Data Scientist and Computational Linguist specializing in Greek digital media and political discourse. Your task is to perform a deep-structure ideological analysis of Greek news articles.

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
{"reasoning": "Το άρθρο χρησιμοποιεί έντονα φορτισμένους όρους όπως 'νεοφιλελεύθερη λαίλαπα' και εστιάζει αποκλειστικά σε ανακοινώσεις συνδικάτων χωρίς να παραθέτει την κυβερνητική θέση, γεγονός που υποδηλώνει σαφή αριστερή/αντισυστημική απόκλιση.", "primary_entities": ["Κυβέρνηση", "ΓΣΕΕ"], "bias": 0.18}"""


# ──────────────────────────────────────────────
# CORE INFERENCE FUNCTION
# ──────────────────────────────────────────────
def call_vllm(article_text: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article_text}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        response = requests.post(
            VLLM_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return json.dumps({"error": str(e), "reasoning": "", "primary_entities": [], "bias": -1.0})


def safe_parse(raw: str) -> dict:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # model wrapped JSON in prose — extract the first {...} block
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    print(f"[PARSE FAIL] Raw response was:\n{raw}\n")
    return {"error": "parse_failed", "raw": raw, "reasoning": "", "primary_entities": [], "bias": -1.0}


def label_article(article_text: str) -> str:
    raw = call_vllm(article_text)
    parsed = safe_parse(raw)
    return json.dumps(parsed, ensure_ascii=False)


# ──────────────────────────────────────────────
# BATCH PROCESSING (parallel HTTP, no Spark)
# ──────────────────────────────────────────────
def label_batch(articles: list[str]) -> list[dict]:
    results = [None] * len(articles)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(label_article, article): i
            for i, article in enumerate(articles)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = json.loads(future.result())
            except Exception as e:
                results[idx] = {"error": str(e), "bias": -1.0}
    return results


# ──────────────────────────────────────────────
# SPARK INTEGRATION
# ──────────────────────────────────────────────
def label_spark_df(df, text_col: str = "text"):
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import StringType
    label_udf = udf(label_article, StringType())
    return df.withColumn("bias_raw", label_udf(col(text_col))) \
             .withColumn("bias_label", col("bias_raw"))


# ──────────────────────────────────────────────
# STANDALONE TEST (no Spark needed)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    hardcoded_articles = [
        {"title": "Μηχανάκι καρφώθηκε σε κολώνα στη Ρόδο - Νεκρός 35χρονος", "text": "Ρόδος Νεκρός Τροχαίο δυστύχημα Την τελευταία του πνοή άφησε μετά από τροχαίο δυστύχημα ένας άνδρας στη Ρόδο . Ο 35χρονος άνδρας με καταγωγή από τον Αρχάγγελο κινούνταν με το δίκυκλό του επί της Ρόδου – Λίνδου με κατεύθυνση προς την Αφάντου, σύμφωνα με το rodiaki.gr. Κάτω από αδιευκρίνιστες συνθήκες έχασε τον έλεγχο του δικύκλου και καρφώθηκε σε κολώνα φωτισμού, σε διαχωριστική νησίδα. Προανάκριση για τα αίτια του δυστυχήματος διενεργεί το τοπικό αστυνομικό τμήμα."},
        {"title": "Ο Θρίαμβος του Τραμπ: Η Παγκοσμιοποιητική Ελίτ Αναγκάζεται να Υποκλιθεί στον Προστάτη των Εθνών!", "text": "Ο Ντόναλντ Τραμπ, ο αδιαμφισβήτητος ηγέτης του πατριωτικού μετώπου, ανακηρύχθηκε «Πρόσωπο της Χρονιάς» για το 2024 από το περιοδικό Time. Μια πικρή αλήθεια για τα συστημικά ΜΜΕ και τους εθνομηδενιστές που επί χρόνια τον λοιδωρούσαν. Είναι η δεύτερη φορά που ο Τραμπ κατακτά αυτόν τον τίτλο, σηματοδοτώντας όχι απλώς μια επιστροφή, αλλά έναν θρίαμβο της λαϊκής βούλησης απέναντι στην Νέα Τάξη Πραγμάτων."},
        {"title": "Η Νέα Τάξη Πραγμάτων στο τένις: Πώς οι δισεκατομμυριούχοι-αφέντες αγοράζουν την δόξα – Η «σταρ» Emma Navarro, ένα ακόμη προϊόν της ελίτ!", "text": "Η Emma Navarro, η δήθεν νέα «αστέρας» του τένις, δεν είναι παρά ένα ακόμη προϊόν του σάπιου συστήματος που κυριαρχεί στον παγκόσμιο αθλητισμό, μια ωμή απόδειξη της Νέας Τάξης Πραγμάτων που μας επιβάλλουν. Τα συστημικά ΜΜΕ, πιστά στην γραμμή των αφεντικών τους, την πλασάρουν ως «νέο αστέρι», ενώ η αλήθεια είναι μία: η 8η θέση της στην WTA δεν είναι αποτέλεσμα ταλέντου, αλλά της απύθμενης οικονομικής ισχύος της οικογένειάς της."},
        {"title": "Ακόμα προλαβαίνουμε: Δέκα βήματα που μπορούν να αλλάξουν τη διατροφή μας πριν από το τέλος του χρόνου", "text": "Οχρόνος τρέχει και η αντίστροφη μέτρηση για το τέλος του χρόνου έχει ήδη ξεκινήσει. Ωστόσο, πριν το 2025 φτάσει, έχουμε μια τελευταία ευκαιρία να κάνουμε αλλαγές που θα μας βοηθήσουν να καλωσορίσουμε τη νέα χρονιά με φρέσκια προοπτική και στόχους. Αν η διατροφή σας είναι κάτι που θα θέλατε να βελτιώσετε έχετε ακόμα χρόνο για να κάνετε μερικά θετικά βήματα πριν καταφτάσει το νέο έτος. Η διατροφολόγος Κλειώ Δημητριάδου προτείνει δέκα απλά βήματα που θα σας βοηθήσουν να πετύχετε τον σκοπό σας – όποιος κι αν είναι αυτός."},
        {"title": "Κορυφαία Σύνοδος Δυνάμεων: Η Ελλάδα στο Έλεος των Παγκόσμιων Παιχνιδιών – Οι Εθνομηδενιστές Κοιμούνται!", "text": "Ενώ ο πλανήτης παρακολουθεί με κομμένη την ανάσα τις κινήσεις των μεγάλων δυνάμεων, οι Ντόναλντ Τραμπ και Βλαντίμιρ Πούτιν, οι δύο ισχυρότεροι ηγέτες του κόσμου, συνομίλησαν τηλεφωνικά, επιβεβαιώνοντας τη Μόσχα. Την ίδια ώρα, η πατρίδα μας, η Ελλάδα, σέρνεται από τους εθνομηδενιστές και τους πράκτορες της Νέας Τάξης Πραγμάτων, που είναι απασχολημένοι με την εθνική μειοδοσία και την υποδοχή λαθρομεταναστών."},
        {"title": "Αναστολή κομματικής ιδιότητας της Αναστασίας Χατζηδάκη για την υπόθεση ΟΠΕΚΑ", "text": "Αναταράξεις στο για και τα φερόμενο «μαϊμού επιδόματα» ύψους 1,8 εκατ. ευρώ. Ο γραμματέας της ΚΠΕ Ανδρέας Σπυρόπουλος, μετά την επιστολή που έλαβε από το μέλος του ΠΑΣΟΚ , κινεί τις διαδικασίες για την έως ότου διαλευκανθεί η υπόθεση. Η ίδια, αναφορικά με τα δημοσιεύματα που την εμπλέκουν και τον συστημικό έλεγχο που διενεργήθηκε στον ΟΠΕΚΑ αναφέρει τα εξής:"},
        {"title": "Καταγγελία Μαμουλάκη για ΒΟΑΚ: «Χάθηκαν 38.000.000 ευρώ και είναι μόνο η αρχή»", "text": "Πολιτική σύγκρουση για το έργο του ΒΟΑΚ στο τμήμα Νεάπολη – Άγιος Νικόλαος, αφού με υπουργική απόφαση προχωρά η απένταξη 38 εκατ. ευρώ ευρωπαϊκών πόρων από το Ταμείο Ανάκαμψης, ανατρέποντας τον σχεδιασμό. «Η απόφαση απένταξης 38.000.000 ευρώ από το Ταμείο Ανάκαμψης για τον ΒΟΑΚ, δεν είναι ένα απλό γεγονός. Είναι η ζημιά που θα πληρώσουμε όλοι, με αποκλειστική ευθύνη της κυβέρνησης της ΝΔ.»"},
        {"title": "Κυκλοφοριακές ρυθμίσεις την Κυριακή στην Λ. Ποσειδώνος λόγω αγώνα δρόμου", "text": "Λεωφόρος Ποσειδώνος Κυκλοφοριακές Ρυθμίσεις Αγώνας Δρόμου Λόγω διεξαγωγής αγώνα με την επωνυμία «Run the lake Vouliagmeni», την Κυριακή (15/12) και κατά τις ώρες 07:00 έως 12:30, θα πραγματοποιηθεί προσωρινή διακοπή της κυκλοφορίας των οχημάτων, επί της Λ. Ποσειδώνος. Η ΕΛΑΣ κάνει έκκληση στους οδηγούς να αποφύγουν τη διέλευση των οχημάτων τους στο παραπάνω οδικό δίκτυο κατά τα αναφερόμενα χρονικά διαστήματα."},
        {"title": "Με «γιούχες» αποχαιρέτησαν τον Γεωργιάδη από το νοσοκομείο της Σύρου [βίντεο]", "text": "Ξινή του βγήκε του υπουργού Υγείας, Άδωνι Γεωργιάδη, ακόμη μια επικοινωνιακή φιέστα, καθώς στα εγκαίνια της νέα Ογκολογικής Μονάδας του νοσοκομείου της Σύρου ήρθε αντιμέτωπος με μια μεγαλειώδη συγκέντρωση διαμαρτυρίας. Ο Σύλλογος Εργαζομένων Νοσοκομείου Σύρου κατέθεσε στον υπουργό ένα υπόμνημα με τα προβλήματα στελέχωσης του Νοσοκομείου, αλλά ο υπουργός απάντησε «Δεν είναι και τραγική η κατάσταση. Και πάλι καλά να λέτε»."},
        {"title": "Το εστιατόριο Masa με το αλησμόνητο σούσι των 1000 δολαρίων το άτομο ανοίγει στο Λονδίνο", "text": "Η φινέτσα, η απλότητα και ο απόλυτος σεβασμός στην πρώτη ύλη που φθάνει από τη διάσημη ιχθυαγορά του Tsukiji του Τόκιο στα εστιατόριά του χαρακτηρίζουν τη δουλειά του Masa Takayama. Το όνομα του οποίου φιγουράρει στην ελίτ της γαστρονομίας, αφού έχει κατακτήσει την απόλυτη διάκριση: τα 3 αστέρια Michelin."},
    ]
 
    test_articles = [
        f"ΤΙΤΛΟΣ: {a['title']}\nΚΕΙΜΕΝΟ: {a['text']}"
        for a in hardcoded_articles
    ]
 
    print(f"Sending {len(test_articles)} articles to vLLM server at {VLLM_URL}\n")
    results = label_batch(test_articles)
 
    print("=" * 60)
    for i, result in enumerate(results):
        if "error" not in result:
            print(f"✅ [{i+1}] bias={result.get('bias', 'N/A')} | entities={result.get('primary_entities', [])}")
            print(f"       reasoning: {result.get('reasoning', '')[:100]}...")
        else:
            print(f"❌ [{i+1}] ERROR: {result.get('error', '')}")
        print()
    print("=" * 60)
    print(f"Done: {sum(1 for r in results if 'error' not in r)}/{len(results)} successful")
 