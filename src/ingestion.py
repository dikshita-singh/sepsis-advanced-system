import pandas as pd
import os

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw sepsis CSV dataset into a Pandas DataFrame.
    """
    print(f"Loading raw data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the dataset file at {file_path}")
        
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data. Shape: {df.shape}")
    return df