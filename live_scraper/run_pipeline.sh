#!/bin/bash
# One scrape pass + export for the Kafka producer (backend/producer.py reads live_scraper/kafka_feed.jsonl).
# For continuous scraping use `python -m live_scraper loop` instead (or call this from cron).
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY -m live_scraper run              # exits 1 if a health check fails; the export still runs
$PY -m live_scraper export --hours 24
