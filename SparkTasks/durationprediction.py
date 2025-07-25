#exception to include in the analysis due to timestamp datatype
#This script predicts the duration of tasks based on various resource usage metrics using Spark MLlib.

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("DurationPrediction").getOrCreate()

df = spark.read.csv("E:/Project_LAB/Preprocess_DATASET/MapReduce/Datasetbigdata.csv", header=True, inferSchema=True).dropna()
df = df.withColumn("duration", df.end_time - df.start_time)

features = ['average_usage_cpu', 'avg_usage_memory', 'maximum_usage_memory', 'maximum_usage',
            'random_sample_usage_memory', 'random_sample_usage_cpu', 'assigned_memory',
            'resource_request', 'cycles_per_instruction', 'memory_accesses_per_instruction',
            'sample_rate', 'cpu_usage_distribution', 'tail_cpu_usage_distribution']

assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(labelCol="duration", featuresCol="features")
model = lr.fit(train_df)
predictions = model.transform(test_df)

for metric in ["rmse", "mae", "r2"]:
    evaluator = RegressionEvaluator(labelCol="duration", predictionCol="prediction", metricName=metric)
    print(f"{metric.upper()}: {evaluator.evaluate(predictions)}")

spark.stop()
