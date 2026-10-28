import pandas as pd

df_raw1 = pd.read_json("articles.json")
df_raw2 = pd.read_json("efsyn.json")

df_raw = pd.concat([df_raw1, df_raw2], ignore_index=True)
print(df_raw)

df_raw.to_json("raw_news.json", orient='split', index=False)
df_raw.to_csv("raw_news.csv", index=False)
