import json
import os
from confluent_kafka import Producer

# Kafka Config
conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)
TOPIC = 'raw-articles'

def delivery_report(err, msg):
    if err is not None:
        print(f"❌ Failed: {err}")
    else:
        print(f"✅ Sent: {msg.topic()} [Partition: {msg.partition()}]")

def send_articles(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            
            # Load line to ensure it's valid JSON
            data = json.loads(line)
            
            # Extract title for the console log
            print(f"Processing: {data.get('title')[:50]}...")

            # Push to Kafka
            producer.produce(
                TOPIC, 
                value=json.dumps(data, ensure_ascii=False).encode('utf-8'), 
                callback=delivery_report
            )
            producer.poll(0)
    
    producer.flush()

if __name__ == "__main__":
    # Adjust path if running from root or scripts folder
    send_articles('datasets/jsonl_demo_mini.jsonl')
