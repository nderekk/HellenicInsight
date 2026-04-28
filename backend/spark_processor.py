import hashlib
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    ArrayType, DoubleType, LongType, StringType, StructField, StructType
)
from dotenv import load_dotenv

load_dotenv()

# vllm_client lives in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_client import label_batch

spark = SparkSession.builder \
    .appName("GreekPoliticsClassifier") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1,"
            "org.mongodb.spark:mongo-spark-connector_2.13:11.0.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Schema of raw articles coming from Kafka (no ai_labels yet)
input_schema = StructType([
    StructField("source", StringType(), True),
    StructField("url",    StringType(), True),
    StructField("title",  StringType(), True),
    StructField("date",   LongType(),   True),
    StructField("text",   StringType(), True),
])

# Schema written to MongoDB (with ai_labels populated by vLLM)
output_schema = StructType([
    StructField("_id",     StringType(), False),
    StructField("source",  StringType(), True),
    StructField("url",     StringType(), True),
    StructField("title",   StringType(), True),
    StructField("content", StringType(), True),
    StructField("date",    LongType(),   True),
    StructField("ai_labels", StructType([
        StructField("reasoning",        StringType(),        True),
        StructField("primary_entities", ArrayType(StringType()), True),
        StructField("bias",             DoubleType(),        True),
    ]), True),
])

raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "raw-articles") \
    .option("startingOffsets", "earliest") \
    .load()

parsed_stream = raw_stream \
    .selectExpr("CAST(value AS STRING) as json_payload") \
    .select(from_json(col("json_payload"), input_schema).alias("article")) \
    .select("article.*")


def write_to_mongo(df, batch_id):
    rows = df.collect()
    if not rows:
        return

    # Format text the same way vllm_client's standalone test does
    texts = [
        f"ΤΙΤΛΟΣ: {r['title']}\nΚΕΙΜΕΝΟ: {r['text']}"
        for r in rows
    ]

    labels = label_batch(texts)

    labeled = []
    for row, label in zip(rows, labels):
        labeled.append({
            "_id":     hashlib.md5((row["url"] or "").encode()).hexdigest(),
            "source":  row["source"],
            "url":     row["url"],
            "title":   row["title"],
            "content": row["text"],
            "date":    row["date"],
            "ai_labels": {
                "reasoning":        label.get("reasoning", ""),
                "primary_entities": label.get("primary_entities", []),
                "bias":             float(label.get("bias", -1.0)),
            },
        })

    labeled_df = spark.createDataFrame(labeled, schema=output_schema)
    labeled_df.write \
        .format("mongodb") \
        .mode("append") \
        .option("connection.uri",  os.getenv("MONGO_URI")) \
        .option("database",        os.getenv("DB_NAME")) \
        .option("collection",      os.getenv("COLLECTION_NAME")) \
        .save()

    print(f"[batch {batch_id}] wrote {len(labeled)} articles to MongoDB")


query = parsed_stream.writeStream \
    .foreachBatch(write_to_mongo) \
    .option("checkpointLocation", "/tmp/spark_checkpoints/articles_to_mongo") \
    .start()

print("Spark is listening on raw-articles → vLLM → MongoDB")
query.awaitTermination()
