from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, expr, struct, lit, to_timestamp,md5
from pyspark.sql.types import StructType, StructField, StringType, LongType

# Start Spark with Kafka and MongoDB connector packages
spark = SparkSession.builder \
    .appName("GreekPoliticsClassifier") \
    .config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1," + 
            "org.mongodb.spark:mongo-spark-connector_2.13:11.0.1") \
    .getOrCreate()
    
spark.sparkContext.setLogLevel("WARN")

#Defined the schema based on the incoming JSONL file
schema = StructType([
    StructField("source", StringType(), True), 
    StructField("url", StringType(), True),
    StructField("title", StringType(), True),
    StructField("date", LongType(), True),     
    StructField("text", StringType(), True)
])

#Connect to Kafka
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "raw-articles") \
    .option("startingOffsets", "earliest") \
    .load()

#Transform: Binary -> String -> JSON -> Columns
parsed_stream = raw_stream.selectExpr("CAST(value AS STRING) as json_payload") \
    .select(from_json(col("json_payload"), schema).alias("article")) \
    .select("article.*")

#Political Classification Logic
classified_stream = parsed_stream.withColumn(
    "political_focus", 
    expr("CASE WHEN text LIKE '%Τραμπ%' THEN 'International/US' ELSE 'General' END")
)

#Align Schema with Mongoose Model
# Simplified selection for debugging
mongodb_stream = classified_stream.select(
    md5(col("url")).alias("_id"), # Generate a unique MongoDB ID
    col("source"),
    col("url"),
    col("title"),
    col("text").alias("content"),
    to_timestamp(expr("CAST(date AS DOUBLE) / 1000")).alias("date"),
    col("political_focus").alias("bias")
)

# 7. Output to MongoDB
# Remember to replace 'your_db_name' with your actual MongoDB database name!
def write_to_mongo(df, batch_id):
    if df.count() > 0:
        df.write \
            .format("mongodb") \
            .mode("append") \
            .option("connection.uri", "mongodb+srv://admin:admin@cluster0.kdfknyi.mongodb.net/") \
            .option("database", "news_database") \
            .option("collection", "articles") \
            .save()

# Use foreachBatch instead of .format("mongodb")
query = mongodb_stream.writeStream \
    .foreachBatch(write_to_mongo) \
    .option("checkpointLocation", "/tmp/spark_checkpoints/articles_to_mongo") \
    .start()

print("🚀 Spark is listening! Processing using foreachBatch...")
query.awaitTermination()