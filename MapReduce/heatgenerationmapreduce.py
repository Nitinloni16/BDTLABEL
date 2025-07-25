import json
import psutil
import time
import csv
import os

# Input MapReduce results file
RESULTS_FILE = 'E:\Project_LAB\Preprocess_DATASET\MapReduce\mapreduce_results.json'

# Output file to store psutil-based heat estimation logs
PSUTIL_LOG = 'mapreduce_psutil_heat_log_from_json.csv'

# Load MapReduce JSON results
def load_results():
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    return data.get('analysis_results', {})

# Estimate heat for each section using psutil
def analyze_with_psutil(results):
    print("🔍 Starting psutil-based heat estimation from JSON results...\n")
    log_rows = []

    process = psutil.Process(os.getpid())

    for key, stats in results.items():
        # Measure pre-execution metrics
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        # Simulate small computation over stats (mimic analytics)
        dummy = sum([
            stats.get('avg_cpu_usage', 0),
            stats.get('avg_memory_usage', 0),
            stats.get('avg_duration', 0),
            stats.get('avg_heat_generation', 0),
            stats.get('total_energy_consumption', 0),
            stats.get('total_heat', 0)
        ]) * 0.0001  # Simulate CPU load

        end_time = time.time()
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / (1024 * 1024)

        duration = end_time - start_time
        avg_cpu = (cpu_before + cpu_after) / 2
        avg_mem = (mem_before + mem_after) / 2

        estimated_heat = avg_cpu * duration  # Simplified heat estimate

        print(f"Key: {key}, Duration: {duration:.4f}s, Avg CPU: {avg_cpu:.2f}%, Heat: {estimated_heat:.4f}")

        log_rows.append({
            "Key": key,
            "Avg_CPU_Usage": round(avg_cpu, 2),
            "Avg_Memory_MB": round(avg_mem, 2),
            "Duration_s": round(duration, 4),
            "Estimated_Heat": round(estimated_heat, 4)
        })

    return log_rows

# Save results to CSV
def save_to_csv(log_rows):
    with open(PSUTIL_LOG, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Key", "Avg_CPU_Usage", "Avg_Memory_MB", "Duration_s", "Estimated_Heat"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n✅ psutil-based heat analysis saved to: {PSUTIL_LOG}")

# Main
if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ {RESULTS_FILE} not found. Please run the MapReduce job first.")
    else:
        results = load_results()
        log_rows = analyze_with_psutil(results)
        save_to_csv(log_rows)
