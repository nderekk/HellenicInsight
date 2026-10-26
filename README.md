# HellenicInsight

> Real-time Greek news bias analysis — powered by a fine-tuned Qwen model with LoRA adapters, streaming through Kafka and Spark into a live React dashboard.

---

## Overview

HellenicInsight is an end-to-end platform that continuously scrapes Greek news articles, classifies their political bias using a fine-tuned Qwen LLM, and surfaces the results on a live frontend dashboard.

Each article is classified with:

| Field | Description |
|---|---|
| **bias** | Float 0–1 (mapped to Left / Center / Right) |
| **reasoning** | The model's plain-language explanation of its bias assessment |
| **primary_entities** | Key political entities mentioned in the article |

---

## Architecture

```
  Live Scraping                Streaming              Inference & Storage
  ─────────────────────────────────────────────────────────────────────────

       Site1.gr ──┐
                  │
                  │
       ........   ├──▶  Scrapy Spider  ──▶  Cleaner  ──▶  Kafka
                  │     (live_scraper/)               (raw-articles)
                  │                                          │
       SiteN.gr ──┘                                         ▼
                                                    Spark Structured
                                                      Streaming
                                                            │
                                                            ▼
                                                    Qwen + LoRA
                                                    (bias inference)
                                                            │
                                                            ▼
                                                        MongoDB
                                                            │
                                                            ▼
                                                  Express REST API
                                                    (backend/)
                                                            │
                                                            ▼
                                                  React Dashboard
                                                    (frontend/)
```

The backend also orchestrates the scraping pipeline automatically via a cron job that runs every hour.

---

## Project Structure

```
HellenicInsight/
│
├── live_scraper/               # RSS/sitemap scraper — how it works: live_scraper/README.md
│   ├── sites.toml              # One block per outlet: feeds, backfill, per-site cleanup
│   ├── scraper.py              # CLI + pipeline (run / loop / backfill / reextract / health / stats / export)
│   ├── net.py · discover.py · extract.py · store.py · health.py
│   ├── run_pipeline.sh         # One pass + export for the Kafka producer
│   ├── kafka_feed.jsonl        # (gitignored) exported articles, ready for Kafka producer
│   └── data/                   # (gitignored) scraper.db + raw HTML
│
├── backend/
│   ├── server.js               # Express API + pipeline orchestrator
│   ├── producer.py             # Kafka producer — publishes kafka_feed.jsonl to raw-articles
│   ├── spark_processor.py      # Spark Structured Streaming — Kafka → inference → MongoDB
│   ├── vllm_client.py          # Client for the vLLM-served Qwen + LoRA model
│   └── qwen_analyzer.py        # On-demand URL analysis service
│
├── frontend/                   # React + Vite + Tailwind + shadcn/ui dashboard
│
├── docs/                       # Architecture, design decisions, pipeline diagram, final presentation
│
├── old-dep/                    # Archive of the research & training phase (not needed to run)
│   ├── 1st_training_dump/      # NeMo SFT splits + trained LoRA weights (qwen_greek_lora_final.tar.gz)
│   ├── datasets/               # Labeled / unlabeled / synthetic training data
│   ├── model_metrics/          # Benchmarks, F1 & confusion heatmap
│   ├── scraper/                # Historical bulk scraper used for dataset collection
│   ├── scripts/                # Labeling (Gemini), conversion & evaluation scripts
│   ├── backend-unused/         # Earlier Mongoose routes/controllers (unused by server.js)
│   └── misc/                   # Early notes and sample files
│
├── docker-compose.yml          # Kafka + Kafka UI
└── requirements.txt
```

---

## Prerequisites

- Python 3.12+
- Node.js 18+
- Docker & Docker Compose
- Java 11+ (for Spark)
- MongoDB instance (local or Atlas)

---

## Steps of Execution

### Step 1 — Configure Environment Variables

Create a `.env` file in `backend/`:

```
MONGO_URI=mongodb://localhost:27017
DB_NAME=hellenic_insight
COLLECTION_NAME=articles
PORT=5000
```

---

### Step 2 — Start Kafka

```bash
docker-compose up -d
```

- Kafka broker: `localhost:9092`
- Kafka UI: `http://localhost:8080`

---

### Step 3 — Start the Spark Processor

The Spark job reads from Kafka and writes classified articles to MongoDB.

```bash
cd backend

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1,org.mongodb.spark:mongo-spark-connector_2.13:11.0.1 \
  spark_processor.py
```

---

### Step 4 — Start the Backend

```bash
cd backend
npm install
node server.js
```

The backend:
- Serves the REST API at `http://localhost:5000/api/articles`
- Runs the scrape → clean → produce pipeline automatically every hour at :45

---

### Step 5 — Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard available at `http://localhost:5173`.

---

### Manual Pipeline Run (optional)

How the scraper works (feeds, sitemaps, storage, health checks): [`live_scraper/README.md`](live_scraper/README.md).

To trigger the scraping pipeline manually without waiting for the cron:

```bash
# 1. Scrape + export (from the repo root; deps: pip install -r live_scraper/requirements.txt)
python -m live_scraper run
python -m live_scraper export --hours 24

# 2. Produce to Kafka
cd backend
python producer.py
```

---

## Frontend Dashboard

The React dashboard:
- **Live feed** — polls `/api/articles` every 10 seconds
- **Political lean meter** — visual Left / Center / Right indicator per article
- **Reasoning log** — the model's plain-language bias explanation
- **Entity tags** — key political figures and parties detected
- **URL analysis** — paste any article URL for on-demand inference

---

## Tech Stack

| Layer | Technology |
|---|---|
| Scraping | Scrapy (SitemapSpider, incremental state) |
| Streaming | Apache Kafka |
| Processing | Apache Spark Structured Streaming |
| LLM | Qwen + LoRA fine-tuned on Greek media (NeMo SFT) |
| Labeling (training) | Google Gemini 2.5 Flash via Vertex AI |
| Storage | MongoDB |
| Backend | Node.js + Express |
| Frontend | React + Vite + Tailwind CSS + shadcn/ui |
| Infrastructure | Docker Compose |

