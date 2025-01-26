def normalize_data(data):
    """
    Normalize dataset for entropy analysis.
    """
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

def calculate_collision_rate(hash_fn, data):
    """
    Calculate the collision rate for a hash function applied to the dataset.
    """
    hashes = [hash_fn(row) for row in data]
    unique_hashes = len(set(hashes))
    return 1 - unique_hashes / len(data)
