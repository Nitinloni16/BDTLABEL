# Priority_mapper class for processing priority data
from collections import defaultdict
def map_priority(row):
    prio = row.get('priority', '').strip()
    if not prio:
        return []
    cpu = float(row.get('average_usage_cpu', 0) or 0)
    failed = row.get('failed', '').strip().lower() in ['true', '1', 'yes']
    heat = cpu * 1.8
    return [('priority_' + prio, (1, cpu, heat, int(failed)))]
# Priority_reducer class for aggregating priority data
from collections import defaultdict
def reduce_priority(data):
    result = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for key, (count, cpu, heat, failed) in data:
        result[key][0] += count
        result[key][1] += cpu
        result[key][2] += heat
        result[key][3] += failed
    return {
        k: {
            'tasks': v[0],
            'avg_cpu': round(v[1]/v[0], 2),
            'avg_heat': round(v[2]/v[0], 2),
            'failure_rate': round(v[3]/v[0]*100, 2)
        } for k, v in result.items()
    }

# Priority_driver class for orchestrating the mapping and reducing process
import csv
from multiprocessing import Pool
from priority_mapper import map_priority
from priority_reducer import reduce_priority
def run_priority_analysis():
    with open('Datasetbigdata.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for _, row in zip(range(1000), reader)]
    with Pool() as pool:
        mapped = pool.map(map_priority, rows)
    flat_mapped = [item for sublist in mapped for item in sublist]
    reduced = reduce_priority(flat_mapped)
    for prio, stats in reduced.items():
        print(f"{prio}: {stats}")
if __name__ == "__main__":
    run_priority_analysis()

