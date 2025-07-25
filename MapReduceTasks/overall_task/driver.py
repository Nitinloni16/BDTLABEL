# Overalltask_mapper class for processing overall task data
def map_overall(row):
    cpu = float(row.get('average_usage_cpu', 0) or 0)
    memory = float(row.get('assigned_memory', 0) or 0)
    duration = float(row.get('end_time', 0) or 0) - float(row.get('start_time', 0) or 0)
    failed = row.get('failed', '').strip().lower() in ['true', '1', 'yes']
    heat = cpu * (duration / 3600) * 10
    energy = (cpu * memory * duration) / 1_000_000
    return [('overall', (1, cpu, memory, heat, energy, duration, int(failed)))]
# Overalltask_reducer class for aggregating overall task data
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
#overalltask_driver class for orchestrating the mapping and reducing process
import csv
from multiprocessing import Pool
from overall_mapper import map_overall
from machine_reducer import reduce_machine_features  # reused
def run_overall():
    with open('Datasetbigdata.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for _, row in zip(range(1000), reader)]
    with Pool() as pool:
        mapped = pool.map(map_overall, rows)
    flat_mapped = [item for sublist in mapped for item in sublist]
    reduced = reduce_machine_features(flat_mapped)
    print("Overall Summary:", reduced['overall'])
if __name__ == "__main__":
    run_overall()
