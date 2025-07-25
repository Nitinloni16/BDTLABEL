# cloud_reducer.py

from statistics import mean

def reduce_sum(values):
    return sum(values)

def reduce_count(values):
    return len(values)

def reduce_avg(values):
    return mean(values) if values else 0

def reduce_list(values):
    return list(values)

def reduce_max(values):
    return max(values)

def reduce_min(values):
    return min(values)
