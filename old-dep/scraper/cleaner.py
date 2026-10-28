import pandas as pd

df_raw = pd.read_json("raw_news.json", orient="split")

text = df_raw["text"]

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

def general_cleaner(raw_text):
    if not raw_text or pd.isna(raw_text):
        return None
    
    clean_text = str(raw_text)
    
    extra_news_marker = 'Ειδήσεις σήμερα:'
    if extra_news_marker in raw_text:
        clean_text = clean_text.split(extra_news_marker)[0]
        
    extra_news_marker2 = 'Ειδήσεις σήμερα'
    if extra_news_marker2 in raw_text:
        clean_text = clean_text.split(extra_news_marker2)[0]
        
    html_vids = " To view this video please enable JavaScript, and consider upgrading to a web browser that supports HTML5 video"
    if html_vids in raw_text:
        clean_text = clean_text.replace(html_vids, "")
        
    return clean_text.strip()

# clean protothema
df_raw.loc[df_raw["source"] == "protothema.gr", 'text'] = df_raw.loc[df_raw["source"] == "protothema.gr", 'text'].apply(clean_protothema_text)
df_raw['text'] = df_raw['text'].apply(general_cleaner)

# sanitize na values
df = df_raw.dropna(subset=['title', 'date', 'url'])

# remove clickbaity articles smaller that 150 characters
df= df[df['text'].str.len() > 150 ]

df['date'] = df['date'].str.slice(0, 16)
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')

df['year'] = df['date'].dt.year
df = df[df['year'].isin([2021, 2022, 2023, 2024, 2025, 2026])]

print(df)

TOTAL_SAMPLES = 6000
PER_SOURCE = TOTAL_SAMPLES//3 # 2k per source

sampled_indices = []

for source_name, group in df.groupby('source'):
    n = min(PER_SOURCE, len(group))
    print(f"Sampling {n} from {source_name}...")
    
    # Grab just the original row indexes for our sample
    sampled_indices.extend(group.sample(n=n, random_state=42).index)
    
final_dataset = df.loc[sampled_indices].sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n--- FINAL DATASET DISTRIBUTION ---")
print(final_dataset['source'].value_counts())
print(f"Total articles ready for labeling: {len(final_dataset)}")

final_dataset[['source', 'url', 'title', 'date', 'text']].to_json('../datasets/final_unlabeled_dataset.json', orient='records', force_ascii=False, indent=2)
# final_dataset[['source', 'url', 'title', 'date', 'text']][:50].to_csv('final_unlabeled_dataset.csv')


print("\nSuccess: 'final_unlabeled_dataset.json' created!")