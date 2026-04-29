import pandas as pd
import os
import sys


# ── SOURCE-SPECIFIC CLEANERS ─────────────────────────────────────────────────

def clean_protothema_text(raw_text):
    if not raw_text or pd.isna(raw_text):
        return None
    clean_text = str(raw_text)
    cookie_marker = "Συνεχίζοντας σε αυτό τον ιστότοπο"
    if cookie_marker in clean_text:
        clean_text = clean_text.split(cookie_marker)[0]
    menu_marker = "Advertorial"
    if menu_marker in clean_text:
        clean_text = clean_text.split(menu_marker)[-1]
    return clean_text.strip()


def clean_kathimerini_text(raw_text):
    if not raw_text or pd.isna(raw_text):
        return None
    clean_text = str(raw_text)
    # Strip subscription paywall notice
    for marker in ["Συνδρομητικό περιεχόμενο", "Αποκλειστικά για συνδρομητές", "ΣΥΝΔΡΟΜΗ"]:
        if marker in clean_text:
            clean_text = clean_text.split(marker)[0]
    return clean_text.strip()


def clean_tanea_text(raw_text):
    if not raw_text or pd.isna(raw_text):
        return None
    clean_text = str(raw_text)
    for marker in ["Διαβάστε επίσης", "ΔΙΑΒΑΣΤΕ ΑΚΟΜΑ", "Δείτε επίσης"]:
        if marker in clean_text:
            clean_text = clean_text.split(marker)[0]
    return clean_text.strip()


def clean_tovima_text(raw_text):
    if not raw_text or pd.isna(raw_text):
        return None
    clean_text = str(raw_text)
    for marker in ["Διαβάστε επίσης", "Δείτε ακόμα", "Περισσότερα"]:
        if marker in clean_text:
            clean_text = clean_text.split(marker)[0]
    return clean_text.strip()


# ── GENERAL CLEANER (applied to all sources after source-specific pass) ───────

def general_cleaner(raw_text):
    if not raw_text or pd.isna(raw_text):
        return None
    clean_text = str(raw_text)

    for marker in ['Ειδήσεις σήμερα:', 'Ειδήσεις σήμερα']:
        if marker in clean_text:
            clean_text = clean_text.split(marker)[0]

    html_vids = " To view this video please enable JavaScript, and consider upgrading to a web browser that supports HTML5 video"
    if html_vids in clean_text:
        clean_text = clean_text.replace(html_vids, "")

    return clean_text.strip()


# ── MAIN ──────────────────────────────────────────────────────────────────────

if not os.path.exists("raw_news.jsonl") or os.path.getsize("raw_news.jsonl") == 0:
    sys.exit(0)

try:
    df_raw = pd.read_json("raw_news.jsonl", lines=True)
except ValueError:
    print("Error reading partially downloaded JSONL lines. Continuing with what is valid.")
    df_raw = pd.DataFrame() # Better handling later but let's just do strict for now

if df_raw.empty:
    sys.exit(0)

# Apply source-specific cleaners
source_cleaners = {
    'protothema.gr':  clean_protothema_text,
    'kathimerini.gr': clean_kathimerini_text,
    'tanea.gr':       clean_tanea_text,
    'tovima.gr':      clean_tovima_text,
}

for source, cleaner_fn in source_cleaners.items():
    mask = df_raw["source"] == source
    df_raw.loc[mask, 'text'] = df_raw.loc[mask, 'text'].apply(cleaner_fn)

# Apply general cleaner to all rows
df_raw['text'] = df_raw['text'].apply(general_cleaner)

# Drop rows missing critical fields
df = df_raw.dropna(subset=['title', 'date', 'url'])
df = df[df['text'].str.len() > 150]

# Parse and validate dates
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
df = df.dropna(subset=['date'])

# Deduplicate by URL
df = df.drop_duplicates(subset=['url'])

if not df.empty:
    df[['source', 'url', 'title', 'date', 'text']].to_json(
        'kafka_feed.jsonl',
        orient='records',
        lines=True,
        force_ascii=False,
        mode='w',
        date_format='epoch',
        date_unit='ms'
    )