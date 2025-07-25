# its not used for the analysis purpose using ML models.
# This script predicts energy consumption based on various resource usage metrics using Spark MLlib.
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import os

# Initialize Spark
spark = SparkSession.builder.appName("EnergyPrediction").getOrCreate()

# Load and preprocess
df = spark.read.csv("E:/Project_LAB/Preprocess_DATASET/MapReduce/Datasetbigdata.csv", header=True, inferSchema=True).dropna()

# Feature engineering
df = df.withColumn("duration", col("end_time") - col("start_time"))
df = df.withColumn("energy", col("average_usage_cpu") * col("duration") * 1e-6)

# Feature columns
features = ['average_usage_cpu', 'avg_usage_memory', 'maximum_usage_memory', 'maximum_usage',
            'random_sample_usage_memory', 'random_sample_usage_cpu', 'assigned_memory', 'duration',
            'resource_request', 'cycles_per_instruction', 'memory_accesses_per_instruction',
            'sample_rate', 'cpu_usage_distribution', 'tail_cpu_usage_distribution']

# Assemble features
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

# Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Model training
lr = LinearRegression(labelCol="energy", featuresCol="features")
model = lr.fit(train_df)
predictions = model.transform(test_df)

# Ensure output directory exists
os.makedirs("artifacts", exist_ok=True)
output_file = "artifacts/energy_prediction_metrics.txt"

# Write metrics to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("⚡ Energy Prediction Evaluation Metrics:\n\n")
    for metric in ["rmse", "mae", "r2"]:
        evaluator = RegressionEvaluator(labelCol="energy", predictionCol="prediction", metricName=metric)
        value = evaluator.evaluate(predictions)
        f.write(f"{metric.upper()}: {value:.4f}\n")

print(f"✅ Evaluation metrics saved to {output_file}")
spark.stop()
