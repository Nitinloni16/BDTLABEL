# Machine_mapper class for processing machine data
def map_machine_features(row):
    machine_id = row.get('machine_id', '').strip()
    if not machine_id:
        return []
    cpu_usage = float(row.get('average_usage_cpu', 0) or 0)
    memory = float(row.get('assigned_memory', 0) or 0)
    duration = float(row.get('end_time', 0) or 0) - float(row.get('start_time', 0) or 0)
    duration = max(duration, 0)
    failed = row.get('failed', '').strip().lower() in ['true', '1', 'yes']
    heat = cpu_usage * (duration / 3600) * 10
    energy = (cpu_usage * memory * duration) / 1_000_000
    return [('machine_' + machine_id, (1, cpu_usage, memory, heat, energy, duration, int(failed)))]

#Machine_reducer class for aggregating machine data
from collections import defaultdict
def reduce_machine_features(mapped_data):
    summary = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
    for key, (count, cpu, memory, heat, energy, duration, failed) in mapped_data:
        summary[key][0] += count
        summary[key][1] += cpu
        summary[key][2] += memory
        summary[key][3] += heat
        summary[key][4] += energy
        summary[key][5] += duration
        summary[key][6] += failed
    return {
        k: {
            'tasks': v[0],
            'avg_cpu': round(v[1]/v[0], 2),
            'avg_memory': round(v[2]/v[0], 2),
            'avg_heat': round(v[3]/v[0], 2),
            'total_energy': round(v[4], 2),
            'failure_rate': round(v[6]/v[0]*100, 2)
        } for k, v in summary.items()
    }
#Machine_driver class for orchestrating the mapping and reducing process
import csv
from multiprocessing import Pool
from machine_mapper import map_machine_features
from machine_reducer import reduce_machine_features

def run_machine_analysis():
    with open('Datasetbigdata.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for _, row in zip(range(1000), reader)]

    with Pool() as pool:
        mapped = pool.map(map_machine_features, rows)
    flat_mapped = [item for sublist in mapped for item in sublist]

    reduced = reduce_machine_features(flat_mapped)
    for machine, stats in reduced.items():
        print(f"{machine}: {stats}")

if __name__ == "__main__":
    run_machine_analysis()
