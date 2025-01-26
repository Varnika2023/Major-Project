import pandas as pd
import numpy as np

def load_sample_data(filepath):
    """
    Load and preprocess data from the CSV file.
    """
    data = pd.read_csv(filepath, header=None)

    # Rename columns for easier access
    data.columns = ["id", "title", "url", "num_points", "num_comments", "author", "created_at"]

    # Fill missing values in numerical columns with 0
    data["num_points"] = pd.to_numeric(data["num_points"], errors="coerce").fillna(0).astype(int)
    data["num_comments"] = pd.to_numeric(data["num_comments"], errors="coerce").fillna(0).astype(int)

    # Select relevant columns for entropy analysis
    relevant_data = data[["num_points", "num_comments"]].values

    return relevant_data
