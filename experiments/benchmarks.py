from entropy_learned_hashing.utils import calculate_collision_rate

def benchmark_hashing(model, data):
    """
    Measure the collision rate and hashing performance of the model.
    """
    collision_rate = calculate_collision_rate(model.hash_function, data)
    return {"collision_rate": collision_rate}
