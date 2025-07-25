from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("HeatPrediction").getOrCreate()

# Load and clean data
df = spark.read.csv("E:/Project_LAB/Preprocess_DATASET/MapReduce/Datasetbigdata.csv", header=True, inferSchema=True).dropna()

# Add duration column first
df = df.withColumn("duration", col("end_time") - col("start_time"))

# Then safely add heat column
df = df.withColumn("heat", (col("average_usage_cpu") * 100) + 10)

# Feature selection
features = ['average_usage_cpu', 'avg_usage_memory', 'maximum_usage_memory', 'maximum_usage',
            'random_sample_usage_memory', 'random_sample_usage_cpu', 'assigned_memory', 'duration',
            'resource_request', 'cycles_per_instruction', 'memory_accesses_per_instruction',
            'sample_rate', 'cpu_usage_distribution', 'tail_cpu_usage_distribution']

# Assemble features
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)

# Train-test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LinearRegression(labelCol="heat", featuresCol="features")
model = lr.fit(train_df)
predictions = model.transform(test_df)

# Evaluate using multiple metrics
for metric in ["rmse", "mae", "r2"]:
    evaluator = RegressionEvaluator(labelCol="heat", predictionCol="prediction", metricName=metric)
    print(f"{metric.upper()}: {evaluator.evaluate(predictions)}")

# Stop Spark session
spark.stop()
