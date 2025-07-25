from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("SchedulingClassPrediction").getOrCreate()

df = spark.read.csv("E:/Project_LAB/Preprocess_DATASET/MapReduce/Datasetbigdata.csv", header=True, inferSchema=True).dropna()
df = df.withColumn("duration", df.end_time - df.start_time)

indexer = StringIndexer(inputCol="scheduling_class", outputCol="scheduling_class_index")
df = indexer.fit(df).transform(df)

features = ['average_usage_cpu', 'avg_usage_memory', 'maximum_usage_memory', 'maximum_usage',
            'random_sample_usage_memory', 'random_sample_usage_cpu', 'assigned_memory', 'duration',
            'resource_request', 'cycles_per_instruction', 'memory_accesses_per_instruction',
            'sample_rate', 'cpu_usage_distribution', 'tail_cpu_usage_distribution']

assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(labelCol="scheduling_class_index", featuresCol="features")
model = rf.fit(train_df)
predictions = model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="scheduling_class_index", predictionCol="prediction")
for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    print(f"{metric.capitalize()}: {evaluator.evaluate(predictions, {evaluator.metricName: metric})}")

spark.stop()
