# Σχεδιαστικές Αποφάσεις, Προβλήματα & Επόμενα Βήματα (Design Decisions)

## 1. AI & Machine Learning Αποφάσεις

### Γιατί Fine-Tuned LLM (LoRA) αντί για BERT (ή κλασσικό classification);
- **Ανάγκη για Reasoning (Επεξήγηση):** Το bias στα ελληνικά media είναι βαθύ και περίπλοκο. Ένα classification model (όπως το BERT) δίνει απλά ένα `Left/Right`. Ένα LLM παρέχει **επεξήγηση του συλλογισμού** του (reasoning) για το πώς το άρθρο "πλαισιώνει" ιδεολογικά (ideological framing) το θέμα.
- **Knowledge Distillation (Απόσταξη Γνώσης):** Αξιοποιήσαμε μεγάλα Foundation Models (ως δασκάλους) για να εξάγουμε τον συλλογισμό τους και να τον "περάσουμε" σε ένα μικρότερο, τοπικό 14B/35B μοντέλο (Qwen). 
- **Τοπικό Inference (Χωρίς API Costs):** Επιτύχαμε λύση πλήρως τοπική, χωρίς εξωτερικές εξαρτήσεις και κόστη API. Αυτό επιτρέπει κλιμάκωση (scalability) και χρήση σε πραγματικές συνθήκες.
- **Low-Rank Adaptation (LoRA):** Αναγκάσαμε το Qwen να λειτουργεί αυστηρά ως classifier που παράγει JSON.

### Τα Αποτελέσματα & Η Επίδραση του Fine-Tuning
Το fine-tuning έδωσε εντυπωσιακά αποτελέσματα (αποδεικνύοντας την ορθότητα της απόφασης):
- Πετύχαμε **~10% F1 boost** στο classification της πολιτικής μεροληψίας.
- Το base μοντέλο (πριν το fine-tuning) είχε την τάση να υποχωρεί στην **ουδετερότητα (neutrality)** όταν αντιμετώπιζε πολύπλοκα δεξιά (Right) άρθρα (είχε κάνει misclassify 17 τέτοια άρθρα ως Center/Neutral).
- Το δικό μας custom μοντέλο "διαγωνιοποίησε" (diagonalized) το confusion matrix, πετυχαίνοντας **~275% βελτίωση** στη συγκεκριμένη κατηγορία!

---

## 2. Αρχιτεκτονικές Αποφάσεις (Data Engineering)

### vLLM για το Inference
Επιλέξαμε vLLM γιατί είναι σχεδιασμένο για **high-throughput concurrent inference** σε HPC hardware (όπως οι B200 GPUs). Υποστηρίζει Prefix Caching και Continuous Batching, που ταίριαζαν απόλυτα στο real-time streaming pipeline μας.

### PySpark & Kafka για το backend
Το **Kafka** εξασφαλίζει την ασύγχρονη διαχείριση των άρθρων από το Scraper. Το **PySpark** αναλαμβάνει το μετασχηματισμό (transform) και τον διαχωρισμό (micro-batching), προετοιμάζοντας τέλεια τα δεδομένα για να τροφοδοτήσουν τις GPUs (vLLM) χωρίς αυτές να "κάθονται", μεγιστοποιώντας την αξιοποίηση του hardware.

---

## 3. Προβλήματα που Αντιμετωπίσαμε (The "Hackathon Mess")

Λόγω του περιορισμένου χρόνου και των συνθηκών του Hackathon, προέκυψαν προκλήσεις και τεχνικά χρέη που εξηγούν την τρέχουσα "μπερδεμένη" μορφή του repo:
1. **Δυσκολίες Υποδομής (Hardware/Network):** 
   - Αργό internet την 1η μέρα καθυστέρησε το testing.
   - Οι NVIDIA B200 GPUs είχαν πολύ σύγχρονες εκδόσεις βιβλιοθηκών, κάνοντας το benchmarking στο GreekMMLU αρκετά "messy".
2. **Scraping Protections:** Πολλά ελληνικά news websites είχαν ανεπτυγμένα bots προστασίας, γεγονός που δυσκόλεψε το single-article scraping και μας ανάγκασε να προσαρμοστούμε με εργαλεία όπως το Scrapy και Crawl4AI.
3. **Frontend/Backend Bugs & Monolithic Design:** 
   - Στον φάκελο `backend/` συνυπάρχουν Node.js (που σερβίρει το React API), Python (Spark processors), και scripts για scraping.
   - Training scripts (`api_label.py`) και datasets είναι ανακατεμένα με το production κώδικα.

---

## 4. Επόμενα Βήματα: Πορεία προς Production

Η εφαρμογή είναι ήδη πολύ κοντά στο να γίνει production-ready, καθώς χρησιμοποιεί σύγχρονο tech stack. Τα άμεσα βήματα οργάνωσης είναι:

### Βήμα 1: Αποσύνδεση και Containerization (Microservices)
Το μονολιθικό Backend πρέπει να σπάσει σε ανεξάρτητα Docker containers:
- **Scraper Service (Python/Crawl4AI):** Ασύγχρονο data extraction στο Kafka.
- **Streaming Service (Spark):** Διαβάζει Kafka, κάνει transform, στέλνει στο vLLM, και κάνει persist στη Mongo.
- **Web API Service (Node.js):** Μιλάει αποκλειστικά με τη MongoDB και το Frontend.

### Βήμα 2: Καθαρισμός του Repository (Refactoring)
- Απομόνωση του MLOps: Δημιουργία φακέλου `research_and_training/` για τα Jupyter notebooks, τα datasets και το labeling (Gemini) pipeline, μακριά από τον production κώδικα.

### Βήμα 3: Cloud Υποδομές & Υποστήριξη
Για το scale-up της πλατφόρμας, κρίσιμη ανάγκη είναι η **συνεχής πρόσβαση σε High-Performance GPU cloud compute** για την αποδοτική εκτέλεση του τοπικού LLM, καθώς και η δυνητική εξασφάλιση seed funding για την κάλυψη των εξόδων server hosting και βάσεων δεδομένων.
