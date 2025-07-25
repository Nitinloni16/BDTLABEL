# Schedulling_mapper class for processing scheduling data
from collections import defaultdict
def map_scheduling_class(row):
    cls = row.get('scheduling_class', '').strip()
    if not cls:
        return []
    cpu = float(row.get('average_usage_cpu', 0) or 0)
    heat = cpu * 2
    failed = row.get('failed', '').strip().lower() in ['true', '1', 'yes']
    return [('class_' + cls, (1, cpu, heat, int(failed)))]

# Schedulling_reducer class for aggregating scheduling data
from collections import defaultdict
def reduce_scheduling_class(data):
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

# Schedulling_driver class for orchestrating the mapping and reducing process
import csv
from multiprocessing import Pool
from scheduling_mapper import map_scheduling_class
from scheduling_reducer import reduce_scheduling_class
def run_sched_class():
    with open('Datasetbigdata.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for _, row in zip(range(1000), reader)]
    with Pool() as pool:
        mapped = pool.map(map_scheduling_class, rows)
    flat_mapped = [item for sublist in mapped for item in sublist]
    reduced = reduce_scheduling_class(flat_mapped)
    for cls, stats in reduced.items():
        print(f"{cls}: {stats}")
if __name__ == "__main__":
    run_sched_class()
