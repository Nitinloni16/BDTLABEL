from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os
# Initialize Spark
spark = SparkSession.builder.appName("CollectionTypePrediction").getOrCreate()
# Load data
df = spark.read.csv("E:/Project_LAB/Preprocess_DATASET/MapReduce/Datasetbigdata.csv", header=True, inferSchema=True).dropna()
df = df.withColumn("duration", df.end_time - df.start_time)
# Encode label
indexer = StringIndexer(inputCol="collection_type", outputCol="collection_type_index")
df = indexer.fit(df).transform(df)
# Feature columns
features = ['average_usage_cpu', 'avg_usage_memory', 'maximum_usage_memory', 'maximum_usage',
            'random_sample_usage_memory', 'random_sample_usage_cpu', 'assigned_memory', 'duration',
            'resource_request', 'cycles_per_instruction', 'memory_accesses_per_instruction',
            'sample_rate', 'cpu_usage_distribution', 'tail_cpu_usage_distribution']
# Vector Assembler
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

# Train/test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train model
rf = RandomForestClassifier(labelCol="collection_type_index", featuresCol="features")
model = rf.fit(train_df)
predictions = model.transform(test_df)

# Prepare output file path
os.makedirs("artifacts", exist_ok=True)
output_file = "artifacts/collection_type_prediction_metrics.txt"

# Write results to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("📊 Collection Type Prediction Evaluation Metrics:\n")
    evaluator = MulticlassClassificationEvaluator(labelCol="collection_type_index", predictionCol="prediction")
    for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        score = evaluator.evaluate(predictions, {evaluator.metricName: metric})
        line = f"{metric.capitalize()}: {score:.4f}\n"
        print(line.strip())  # Optional: print to console
        f.write(line)

# Stop Spark
spark.stop()
