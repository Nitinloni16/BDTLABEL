# cloud_driver.py

from pyspark.sql import SparkSession
from cloud_mapper import *
from cloud_reducer import *

# Initialize Spark session
spark = SparkSession.builder.appName("Cloud Log Analysis").getOrCreate()
sc = spark.sparkContext

# Load the dataset
df = spark.read.option("header", True).csv("data.csv")
rdd = df.rdd.map(lambda row: row.asDict())

# 1. Event Type Count
event_counts = rdd.map(map_event_count) \
                   .groupByKey() \
                   .mapValues(reduce_sum)
event_counts_df = event_counts.toDF(["event", "count"])
event_counts_df.write.option("header", True).csv("output_csv/event_counts")

# 2. Failures per Cluster
cluster_failures = rdd.map(map_cluster_failures) \
                        .groupByKey() \
                        .mapValues(reduce_sum)
cluster_failures_df = cluster_failures.toDF(["cluster", "failures"])
cluster_failures_df.write.option("header", True).csv("output_csv/cluster_failures")

# 3. User-wise Failures
user_failures = rdd.map(map_user_failures) \
                    .groupByKey() \
                    .mapValues(reduce_sum)
user_failures_df = user_failures.toDF(["user", "failures"])
user_failures_df.write.option("header", True).csv("output_csv/user_failures")

# 4. Collection Memory Usage
collection_memory = rdd.map(map_collection_memory) \
                        .groupByKey() \
                        .mapValues(reduce_sum)
collection_memory_df = collection_memory.toDF(["collection_id", "memory"])
collection_memory_df.write.option("header", True).csv("output_csv/collection_memory")

# 5. Event-wise Memory Avg
event_memory_avg = rdd.map(map_event_memory) \
                      .groupByKey() \
                      .mapValues(reduce_avg)
event_memory_avg_df = event_memory_avg.toDF(["event", "avg_memory"])
event_memory_avg_df.write.option("header", True).csv("output_csv/event_memory_avg")

# 6. Failures by Priority
priority_failures = rdd.map(map_priority_failure) \
                      .groupByKey() \
                      .mapValues(reduce_avg)
priority_failures_df = priority_failures.toDF(["priority", "avg_failures"])
priority_failures_df.write.option("header", True).csv("output_csv/priority_failures")

# 7. Duration per Collection
collection_duration = rdd.map(map_collection_duration) \
                          .groupByKey() \
                          .mapValues(reduce_avg)
collection_duration_df = collection_duration.toDF(["collection_id", "avg_duration"])
collection_duration_df.write.option("header", True).csv("output_csv/collection_duration")

# 8. Logs per User
user_logs = rdd.map(map_user_logs) \
               .groupByKey() \
               .mapValues(reduce_sum)
user_logs_df = user_logs.toDF(["user", "log_count"])
user_logs_df.write.option("header", True).csv("output_csv/user_logs")

# 9. Constraint Usage
constraint_usage = rdd.map(map_constraint_usage) \
                     .groupByKey() \
                     .mapValues(reduce_sum)
constraint_usage_df = constraint_usage.toDF(["constraint", "count"])
constraint_usage_df.write.option("header", True).csv("output_csv/constraint_usage")

# 10. Failures per Scheduling Class
schedclass_failures = rdd.map(map_schedclass_failures) \
                         .groupByKey() \
                         .mapValues(reduce_sum)
schedclass_failures_df = schedclass_failures.toDF(["scheduling_class", "failures"])
schedclass_failures_df.write.option("header", True).csv("output_csv/schedclass_failures")

# Stop Spark session
spark.stop()
