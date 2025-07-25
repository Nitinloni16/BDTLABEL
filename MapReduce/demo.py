import csv
import json
from collections import defaultdict
from multiprocessing import Pool
from datetime import datetime
import psutil
import os
import time
import pandas as pd

CSV_FILE = 'Datasetbigdata.csv'
ROW_LIMIT = 1000  # Change as needed
RESULTS_FILE = 'mapreduce_results.json'
HEAT_LOG_FILE = 'mapreduce_psutil_heat_log.csv'

heat_trace = []  # For psutil-based system metrics

#Confirm column names
def print_csv_headers():
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        print("Headers in CSV file:", header)
        return header

#MAP FUNCTION
def map_features(row):
    try:
        start_proc = time.time()
        process = psutil.Process(os.getpid())
        cpu_percent = psutil.cpu_percent(interval=None)
        mem_used = process.memory_info().rss / (1024 * 1024)

        machine_id = row.get('machine_id', '').strip()
        collection_id = row.get('collection_id', '').strip()
        scheduling_class = row.get('scheduling_class', '').strip()
        cpu_usage_dist = float(row.get('cpu_usage_distribution', 0) or 0)
        average_usage_cpu = float(row.get('average_usage_cpu', 0) or 0)
        avg_usage_memory = float(row.get('avg_usage_memory', 0) or 0)
        maximum_usage_memory = float(row.get('maximum_usage_memory', 0) or 0)
        maximum_usage = float(row.get('maximum_usage', 0) or 0)
        random_sample_usage_cpu = float(row.get('random_sample_usage_cpu', 0) or 0)
        random_sample_usage_memory = float(row.get('random_sample_usage_memory', 0) or 0)
        assigned_memory = float(row.get('assigned_memory', 0) or 0)
        page_cache_memory = float(row.get('page_cache_memory', 0) or 0)
        cycles_per_instruction = float(row.get('cycles_per_instruction', 0) or 0)
        memory_accesses_per_instruction = float(row.get('memory_accesses_per_instruction', 0) or 0)
        start_time = float(row.get('start_time', 0) or 0)
        end_time = float(row.get('end_time', 0) or 0)
        duration = end_time - start_time if end_time > start_time else 0
        priority = row.get('priority', '').strip()
        collection_type = row.get('collection_type', '').strip()
        event = row.get('event', '').strip()
        failed = row.get('failed', '').strip()
        if not machine_id:
            return []
        memory_access_factor = 1 + (memory_accesses_per_instruction / 10.0) if memory_accesses_per_instruction > 0 else 1.0
        duration_factor = min(abs(duration) / 3600.0, 1.0) if duration != 0 else 0.1
        cpu_metrics = [average_usage_cpu, random_sample_usage_cpu, cpu_usage_dist]
        effective_cpu_usage = sum(m for m in cpu_metrics if m > 0) / len([m for m in cpu_metrics if m > 0]) if any(m > 0 for m in cpu_metrics) else 0
        memory_metrics = [avg_usage_memory, maximum_usage_memory, random_sample_usage_memory]
        effective_memory_usage = sum(m for m in memory_metrics if m > 0) / len([m for m in memory_metrics if m > 0]) if any(m > 0 for m in memory_metrics) else 0
        estimated_heat = (effective_cpu_usage * memory_access_factor * duration_factor * 100) + (cycles_per_instruction * 0.1)
        energy_consumption = (effective_cpu_usage * assigned_memory * abs(duration)) / 1_000_000 if assigned_memory > 0 else 0
        results = []
        results.append(('machine_' + machine_id, (1, effective_cpu_usage, maximum_usage, assigned_memory,
                                                  estimated_heat, energy_consumption, duration, effective_memory_usage,
                                                  1 if failed.lower() in ['true', '1', 'yes'] else 0)))
        if scheduling_class:
            results.append(('class_' + scheduling_class, (1, effective_cpu_usage, maximum_usage, assigned_memory,
                                                          estimated_heat, energy_consumption, duration, effective_memory_usage,
                                                          1 if failed.lower() in ['true', '1', 'yes'] else 0)))
        if collection_type:
            results.append(('collection_type_' + collection_type, (1, effective_cpu_usage, maximum_usage, assigned_memory,
                                                                   estimated_heat, energy_consumption, duration, effective_memory_usage,
                                                                   1 if failed.lower() in ['true', '1', 'yes'] else 0)))
        if priority:
            results.append(('priority_' + priority, (1, effective_cpu_usage, maximum_usage, assigned_memory,
                                                     estimated_heat, energy_consumption, duration, effective_memory_usage,
                                                     1 if failed.lower() in ['true', '1', 'yes'] else 0)))
        results.append(('overall', (1, effective_cpu_usage, maximum_usage, assigned_memory,
                                   estimated_heat, energy_consumption, duration, effective_memory_usage,
                                   1 if failed.lower() in ['true', '1', 'yes'] else 0)))

        end_proc = time.time()
        duration_proc = end_proc - start_proc
        sys_estimated_heat = cpu_percent * duration_proc
        heat_trace.append({
            'RowKey': machine_id or collection_id or scheduling_class,
            'CPU_Usage': cpu_percent,
            'Memory_MB': mem_used,
            'Duration_s': round(duration_proc, 4),
            'Estimated_Heat': round(sys_estimated_heat, 4)
        })

        return results
    except Exception as e:
        print(f" Error mapping row: {e}")
        return []

# REDUCE FUNCTION (unchanged)
def reduce_features(mapped_data):
    summary = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
    for key, (count, avg_cpu, max_cpu, memory, heat, energy, duration, memory_usage, failed) in mapped_data:
        summary[key][0] += count
        summary[key][1] += avg_cpu
        summary[key][2] += max_cpu
        summary[key][3] += memory
        summary[key][4] += heat
        summary[key][5] += energy
        summary[key][6] += duration
        summary[key][7] += memory_usage
        summary[key][8] += failed
    results = {}
    for key, (count, total_avg_cpu, total_max_cpu, total_memory, total_heat, total_energy, total_duration, total_memory_usage, total_failed) in summary.items():
        if count > 0:
            results[key] = {
                'count': count,
                'avg_cpu_usage': round(total_avg_cpu / count, 4),
                'avg_max_cpu': round(total_max_cpu / count, 4),
                'avg_memory': round(total_memory / count, 2),
                'avg_memory_usage': round(total_memory_usage / count, 4),
                'avg_heat_generation': round(total_heat / count, 2),
                'total_energy_consumption': round(total_energy, 2),
                'avg_duration': round(total_duration / count, 2),
                'total_heat': round(total_heat, 2),
                'failure_rate': round((total_failed / count) * 100, 2) if count > 0 else 0,
                'failed_tasks': total_failed
            }
    return results

# Save results to JSON
def save_results_to_json(results, headers, row_count):
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'row_count': row_count,
        'headers': headers,
        'analysis_results': results
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f" Results saved to {RESULTS_FILE}")

# Main function
def run_cloud_mapreduce():
    print(" Starting Enhanced Cloud Workload Heat Analysis MapReduce Program")
    print("=" * 70)
    headers = print_csv_headers()
    with open(CSV_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for i, row in enumerate(reader) if i < ROW_LIMIT]
    if not rows:
        print(" No rows found in the CSV file.")
        return
    print(f" Processing {len(rows)} rows...")
    with Pool() as pool:
        mapped = pool.map(map_features, rows)
    flat_mapped = [item for sublist in mapped for item in sublist if sublist]
    reduced = reduce_features(flat_mapped)
    save_results_to_json(reduced, headers, len(rows))

    # Save psutil-based heat trace
    heat_df = pd.DataFrame(heat_trace)
    heat_df.to_csv(HEAT_LOG_FILE, index=False)
    print(f"✅ psutil-based heat log saved to {HEAT_LOG_FILE}")

    print("\n" + "="*90)
    print(" ENHANCED CLOUD WORKLOAD HEAT ANALYSIS RESULTS")
    print("="*90)
    if 'overall' in reduced:
        overall = reduced['overall']
        print(f"\n OVERALL SUMMARY:")
        print(f"   Total Tasks: {overall['count']}")
        print(f"   Avg CPU Usage: {overall['avg_cpu_usage']}%")
        print(f"   Peak CPU: {overall['avg_max_cpu']}%")
        print(f"   Assigned Memory: {overall['avg_memory']} MB")
        print(f"   Memory Usage: {overall['avg_memory_usage']}%")
        print(f"   Heat Generation: {overall['avg_heat_generation']}°C")
        print(f"   Energy Consumption: {overall['total_energy_consumption']} kWh")
        print(f"   Avg Duration: {overall['avg_duration']} sec")
        print(f"   Total Heat: {overall['total_heat']}°C")
        print(f"   Failure Rate: {overall['failure_rate']}% ({overall['failed_tasks']} failed)")
    print(f"\n TOP 10 MACHINE-WISE ANALYSIS:")
    machine_data = {k: v for k, v in reduced.items() if k.startswith('machine_')}
    top_machines = sorted(machine_data.items(), key=lambda x: x[1]['avg_heat_generation'], reverse=True)[:10]
    for machine_key, stats in top_machines:
        print(f"   {machine_key.replace('machine_', '')}: {stats['count']} tasks, "
              f"Heat: {stats['avg_heat_generation']}°C, Energy: {stats['total_energy_consumption']} kWh, "
              f"Failure: {stats['failure_rate']}%")
    print(f"\n SCHEDULING CLASS ANALYSIS:")
    for class_key, stats in sorted({k: v for k, v in reduced.items() if k.startswith('class_')}.items()):
        print(f"   Class {class_key.replace('class_', '')}: {stats['count']} tasks, "
              f"CPU: {stats['avg_cpu_usage']}%, Heat: {stats['avg_heat_generation']}°C, "
              f"Failure: {stats['failure_rate']}%")
    print(f"\n COLLECTION TYPE ANALYSIS:")
    for type_key, stats in sorted({k: v for k, v in reduced.items() if k.startswith('collection_type_')}.items()):
        print(f"   {type_key.replace('collection_type_', '')}: {stats['count']} tasks, "
              f"Heat: {stats['avg_heat_generation']}°C, Memory: {stats['avg_memory_usage']}%")
    print(f"\n PRIORITY ANALYSIS:")
    for priority_key, stats in sorted({k: v for k, v in reduced.items() if k.startswith('priority_')}.items()):
        print(f"   Priority {priority_key.replace('priority_', '')}: {stats['count']} tasks, "
              f"Heat: {stats['avg_heat_generation']}°C, Failure: {stats['failure_rate']}%")
    print("\n" + "="*90)
    print(" Enhanced Analysis Complete!")
    return reduced

# Step 6: Run program
if __name__ == "__main__":
    run_cloud_mapreduce()
