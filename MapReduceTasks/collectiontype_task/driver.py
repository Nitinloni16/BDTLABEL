#collectiontype_mapper class for processing collection type data
from collections import defaultdict
def map_collection_type(row):
    col_type = row.get('collection_type', '').strip()
    if not col_type:
        return []
    memory_usage = float(row.get('avg_usage_memory', 0) or 0)
    heat = memory_usage * 1.5
    return [('collection_type_' + col_type, (1, memory_usage, heat))]

# collectiontype_reducer class for aggregating collection type data
from collections import defaultdict
def reduce_collection_type(data):
    summary = defaultdict(lambda: [0, 0.0, 0.0])
    for key, (count, memory, heat) in data:
        summary[key][0] += count
        summary[key][1] += memory
        summary[key][2] += heat
    return {
        k: {
            'tasks': v[0],
            'avg_memory_usage': round(v[1]/v[0], 2),
            'avg_heat': round(v[2]/v[0], 2)
        } for k, v in summary.items()
    }

# collectiontype_driver class for orchestrating the mapping and reducing process
import csv
from multiprocessing import Pool
from collection_mapper import map_collection_type
from collection_reducer import reduce_collection_type
def run_collection_type():
    with open('Datasetbigdata.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [row for _, row in zip(range(1000), reader)]
    with Pool() as pool:
        mapped = pool.map(map_collection_type, rows)
    flat_mapped = [item for sublist in mapped for item in sublist]
    reduced = reduce_collection_type(flat_mapped)

    for col_type, stats in reduced.items():
        print(f"{col_type}: {stats}")
        
if __name__ == "__main__":
    run_collection_type()
