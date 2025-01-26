import os
from entropy_learned_hashing.entropy_hashing import EntropyLearnedHashing
from experiments.data_loader import load_sample_data
from experiments.benchmarks import benchmark_hashing

def main():
    # Load and preprocess the dataset
    dataset_path = os.path.join("datasets", "sample_data.csv")
    data = load_sample_data(dataset_path)

    # Initialize and train the hashing model
    model = EntropyLearnedHashing()
    model.train(data)

    # Benchmark the model
    results = benchmark_hashing(model, data)
    print("Benchmark Results:", results)

if __name__ == "__main__":
    main()
