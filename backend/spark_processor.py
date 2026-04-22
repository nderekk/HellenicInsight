from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, expr, struct, lit, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, LongType

# Start Spark with Kafka and MongoDB connector packages
spark = SparkSession.builder \
    .appName("GreekPoliticsClassifier") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1,org.mongodb.spark:mongo-spark-connector_2.13:10.3.0") \
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
mongodb_stream = classified_stream.select(
    # Format source as an object
    struct(
        lit(None).cast("string").alias("id"), 
        col("source").alias("name")
    ).alias("source"),
    
    col("url"), # Pass the URL straight through
    col("title"),
    col("text").alias("content"), # Rename text to content
    
    # Convert Unix timestamp to MongoDB Date
    to_timestamp(col("date") / 1000).alias("date"), 
    
    col("political_focus").alias("bias") # Rename political_focus to bias
)

# 7. Output to MongoDB
# Remember to replace 'your_db_name' with your actual MongoDB database name!
query = mongodb_stream.writeStream \
    .format("mongodb") \
    .option("spark.mongodb.connection.uri", "mongodb://localhost:27017") \
    .option("spark.mongodb.database", "your_db_name") \
    .option("spark.mongodb.collection", "articles") \
    .option("checkpointLocation", "/tmp/spark_checkpoints/articles_to_mongo") \
    .outputMode("append") \
    .start()

print("🚀 Spark is listening! Processing and saving directly to MongoDB...")
query.awaitTermination()