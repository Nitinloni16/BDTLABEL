# cloud_mapper.py

def map_event_count(row):
    return (row['event'], 1)

def map_cluster_failures(row):
    return (row['cluster'], int(row['failed']))

def map_user_failures(row):
    return (row['user'], int(row['failed']))

def map_collection_memory(row):
    return (row['collection_id'], float(row['assigned_memory'] or 0.0))

def map_event_memory(row):
    return (row['event'], float(row['assigned_memory'] or 0.0))

def map_machine_usage(row):
    return (row['machine_id'], row['start_time'])

def map_priority_failure(row):
    return (row['priority'], int(row['failed']))

def map_collection_duration(row):
    return (row['collection_id'], int(row['end_time']) - int(row['start_time']))

def map_user_logs(row):
    return (row['user'], 1)

def map_constraint_usage(row):
    return (row['constraint'], 1)

def map_schedclass_failures(row):
    return (row['scheduling_class'], int(row['failed']))

def map_memory_failure(row):
    return (float(row['assigned_memory'] or 0.0), int(row['failed']))
