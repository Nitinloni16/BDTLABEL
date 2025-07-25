#don't consider it for the analysis purpose using ML models.
# This script classifies the top 10 most frequent machine IDs in a dataset using Spark MLlib.
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Start Spark
spark = SparkSession.builder.appName("Top10MachineIDClassification").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Load and preprocess
df = spark.read.csv("E:/Project_LAB/Preprocess_DATASET/MapReduce/Datasetbigdata.csv", header=True, inferSchema=True).dropna()

# 3. Filter Top 10 most frequent machine_ids
top_ids_df = df.groupBy("machine_id").agg(count("*").alias("count")).orderBy(col("count").desc()).limit(10)
top_ids = [row["machine_id"] for row in top_ids_df.collect()]
df = df.filter(col("machine_id").isin(top_ids))

# 4. Derived column
df = df.withColumn("duration", col("end_time") - col("start_time"))

# 5. Index machine_id as label
indexer = StringIndexer(inputCol="machine_id", outputCol="label")
df = indexer.fit(df).transform(df)

# 6. Feature Engineering
feature_cols = ['average_usage_cpu', 'avg_usage_memory', 'maximum_usage_memory', 'maximum_usage',
                'random_sample_usage_memory', 'random_sample_usage_cpu', 'assigned_memory', 'duration',
                'resource_request', 'cycles_per_instruction', 'memory_accesses_per_instruction',
                'sample_rate', 'cpu_usage_distribution', 'tail_cpu_usage_distribution',
                'priority', 'scheduling_class']

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
df = assembler.transform(df)

scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
df = scaler.fit(df).transform(df)

# 7. Check class balance
print("\n🧮 Class Distribution (Top 10 machine_id):")
df.groupBy("label").count().orderBy("count", ascending=False).show()

# 8. Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 9. Train RandomForest Classifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50)
model = rf.fit(train_df)
predictions = model.transform(test_df)

# 10. Evaluate model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
print("\n📊 Classification Metrics (Top 10 machine_id):")
for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    value = evaluator.evaluate(predictions, {evaluator.metricName: metric})
    print(f"{metric.capitalize()}: {value:.4f}")

# 11. Confusion Matrix
print("\n🔍 Confusion Matrix:")
predictions.select("label", "prediction").crosstab("label", "prediction").show(truncate=False)

# 12. Done
spark.stop()
