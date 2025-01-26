import numpy as np

class EntropyLearnedHashing:
    def __init__(self):
        self.selected_columns = None

    def estimate_entropy(self, data):
        """
        Estimate entropy for each column in the dataset.
        """
        entropy = []
        for col in range(data.shape[1]):
            _, counts = np.unique(data[:, col], return_counts=True)
            probabilities = counts / len(data)
            entropy.append(-np.sum(probabilities * np.log2(probabilities)))
        return np.array(entropy)

    def select_columns(self, data, top_k=2):
        """
        Select the top-k columns with the highest entropy.
        """
        entropy = self.estimate_entropy(data)
        self.selected_columns = np.argsort(entropy)[-top_k:]
        print(f"Selected columns for hashing: {self.selected_columns}")

    def hash_function(self, row):
        """
        Hash a data row using only selected columns.
        """
        if self.selected_columns is None:
            raise ValueError("Columns not selected. Train the model first.")
        return hash(tuple(row[col] for col in self.selected_columns))

    def train(self, data):
        """
        Train the hashing model by selecting columns based on entropy.
        """
        self.select_columns(data)
