from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import os

# Initialize Spark
spark = SparkSession.builder.appName("FailurePrediction").getOrCreate()

# Load Data
df = spark.read.csv("E:/Project_LAB/Preprocess_DATASET/MapReduce/Datasetbigdata.csv", header=True, inferSchema=True).dropna()

# Feature Engineering
df = df.withColumn("duration", col("end_time") - col("start_time"))
df = df.withColumn("heat", (col("average_usage_cpu") * 100) + 10)
df = df.withColumn("energy", col("average_usage_cpu") * col("duration") * 1e-6)

# Feature Columns
features = ['average_usage_cpu', 'avg_usage_memory', 'maximum_usage_memory', 'maximum_usage',
            'random_sample_usage_memory', 'random_sample_usage_cpu', 'assigned_memory', 'duration',
            'resource_request', 'cycles_per_instruction', 'memory_accesses_per_instruction',
            'sample_rate', 'cpu_usage_distribution', 'tail_cpu_usage_distribution']

assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

# Split Dataset
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train Model
rf = RandomForestClassifier(labelCol="failed", featuresCol="features")
model = rf.fit(train_df)
predictions = model.transform(test_df)

# Create output directory
os.makedirs("artifacts", exist_ok=True)
output_file = "artifacts/failure_prediction_metrics.txt"

# Write Evaluation Metrics
with open(output_file, "w", encoding="utf-8") as f:
    f.write("❌ Failure Prediction Evaluation Metrics:\n\n")

    # Binary Evaluation (AUC)
    binary_eval = BinaryClassificationEvaluator(labelCol="failed")
    auc_score = binary_eval.evaluate(predictions)
    f.write(f"AUC: {auc_score:.4f}\n")

    # Multiclass Evaluation
    multi_eval = MulticlassClassificationEvaluator(labelCol="failed", predictionCol="prediction")
    for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        value = multi_eval.evaluate(predictions, {multi_eval.metricName: metric})
        f.write(f"{metric.capitalize()}: {value:.4f}\n")

print(f"✅ Metrics saved to {output_file}")
spark.stop()
