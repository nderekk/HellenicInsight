import json
from datetime import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")

def populate_db():
    if not uri:
        print("Error: MONGO_URI not found in .env file.")
        return

    client = MongoClient(uri)
    db = client[db_name]
    collection = db["articles"]

    # Load your JSON file
    try:
        with open("datasets/labeled_dataset.json", "r", encoding="utf-8") as file:
            data = json.load(file)
            
        for entry in data:
            if "date" in entry:
                entry["date"] = datetime.fromtimestamp(entry["date"] / 1000.0)
                
        print(data[0])
        return

        result = collection.insert_many(data)
        print(f"Successfully inserted {len(result.inserted_ids)} articles into {db_name}.")
        
    except FileNotFoundError:
        print("Error: labeled_dataset.json not found.")
    except Exception as e:
        print(f"Something went wrong: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    populate_db()