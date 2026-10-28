# 🧠 SYSTEM CONTEXT: AI ARCHITECT PIPELINE
**Project:** Real-Time Greek Political Bias Classifier (Knowledge Distillation)
**Event:** AI / HPC (High-Performance Computing) Hackathon
**Hardware Availability:** 3x NVIDIA DGX B200 GPUs (576GB VRAM total)

---

## 👥 TEAM STRUCTURE ("Team Big O No")
1. **AI Architect (Me):** Prompt engineering, LoRA fine-tuning, dataset generation, model evaluation.
2. **Kostas (MLOps/Inference):** GPU allocation, deployment, API endpoints (vLLM).
3. **Vaggos (Data Engineer):** PySpark streaming, micro-batching, Kafka integration.
4. **Elmos (Data Ingestion):** Web scraping (Scrapy), bypassing anti-bot protections.

---

## 🎯 THE MISSION (MVP)
Build a distributed pipeline that scrapes Greek political news in real-time and uses a custom-trained, locally hosted Large Language Model to output a precise ideological bias score on a continuous scalar, as well as a reasoning for the choice and some labels.
* **Scale:** `0.0` (Far-Left) to `1.0` (Far-Right), with `0.5` being Center/Neutral.
* **Output Constraint:** Strict, parseable JSON (e.g., `{"bias": 0.72}`). No conversational text.
* **The "HPC Flex" (Our Pitch):** We did not use a massive 72B model or an expensive API. We used Knowledge Distillation to transfer a frontier model's reasoning into a localized 14B/32B model, optimized via LoRA and deployed via vLLM. It processes Greek text 50x faster and 100x cheaper than enterprise baselines.

---

## ⚙️ ARCHITECTURE & TECH STACK
* **Ingestion:** Scrapy -> Kafka.
* **Transmission:** PySpark Structured Streaming. Handles backpressure, micro-batching (to keep the GPU constantly fed), and ChatML string templating on the CPU to save GPU cycles.
* **Base Model:** Qwen 2.5 (14B or 32B) or KriKri-8B (Greek-native Llama 3 derivative).
* **Fine-Tuning:** LoRA (Low-Rank Adaptation) trained in bfloat16. Forces the model to unlearn "chatbot" behaviors and output pure JSON calibrated to our specific bias scale.
* **Deployment Engine:** **vLLM** *(Pivot from TensorRT-LLM)*.
  * *Why vLLM?* It fits our 1-week hackathon timeframe perfectly. It natively supports multi-LoRA routing (`--enable-lora`), automatic Prefix Caching (so we don't pay compute costs for sending the same System Prompt repeatedly), and continuous batching.

---

## 📊 DATASET QUALITY & MLOPS PROTOCOL
We are using **Knowledge Distillation** (using Gemini as a "Teacher" to label the data for the "Student" LoRA model). The biggest risk is the Teacher hallucinating or the Student cheating. We mitigate this using the following protocols:

### 1. The Ground Truth Paradox (Supreme Court Validation)
* **Problem:** Human validation of complex Greek political nuance is slow and subjective.
* **Solution:** We validate the Gemini Teacher Prompt using Multi-Model Consensus. We sample 20 difficult articles and run them through Gemini 1.5, GPT-4o, and Claude 3.5. If they achieve >90% correlation, we trust Gemini to label the remaining 4,000 articles autonomously.

### 2. Shortcut Learning & Source Leakage
* **Problem:** If we only scrape from 2 sources (e.g., one Left, one Right), the LoRA will become a "Website Classifier" rather than a "Bias Detector." It will memorize metadata, length, or specific journalistic dialects.
* **Solution:** * Add "Anchor Sources" to triangulate bias (e.g., AMNA for absolute `0.5` neutral ground, Kathimerini for Center-Right, EfSyn for Left). Aim for 5+ sources.
  * Conduct a **Blind Test**: Strip out publisher names, locations, and journalist names from the validation set to prove the LoRA is learning pure political rhetoric.

### 3. Cross-Lingual Transfer Learning
* **Strategy:** Augment the Greek dataset with a translated English dataset (e.g., Kaggle Political Bias dataset).
* **Mix Ratio:** ~70% Greek Native / ~30% English Translated.
* **Purpose:** The English data teaches the model the stable "Math of Bias" and logical fallacies (acting as a regularizer), while the Greek data localizes that logic to the Greek media landscape.

---

## 📈 EVALUATION METRICS
We do not use standard classification accuracy. Our custom evaluation script parses the LoRA JSON outputs and calculates the **Mean Absolute Error (MAE)** between the LoRA prediction and the Teacher's Ground Truth.
* *Goal:* MAE < 0.10.

---

## 🚀 IMMEDIATE HACKATHON FOCUS
1. **Day 1:** Lock the MVP Pipeline. Scraper -> Kafka -> Spark -> vLLM -> UI. No feature creep until this loop prints JSON successfully.
2. **Day 2:** HPC Optimization. Maximize vLLM throughput, measure TTFT (Time To First Token), and calculate VRAM savings.
3. **Day 3:** Pitch the architecture, the economics, and the data engineering rigor. Build a simple Bloomberg-style Streamlit dashboard for the visual demo.
