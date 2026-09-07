import pandas as pd
import numpy as np

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up missing ICU data using Forward-Filling.
    """
    print("Starting missing value imputation...")
    
    # 1. Sort data by Patient and Hour to make sure we forward fill chronologically
    # If your dataset doesn't have a Patient_ID, we use the row order grouped by hospital stay indicators
    # For this consolidated Kaggle CSV, we group by 'Patient_ID' if it exists, or fill sequentially
    if 'Patient_ID' in df.columns:
        # Forward fill within each patient's timeline so patient data doesn't bleed into another patient
        df = df.groupby('Patient_ID', group_keys=False).apply(lambda x: x.ffill())
    else:
        df = df.ffill()
        
    print("Forward-filling completed.")
    return df

def drop_empty_columns(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Drops columns that are still missing more than the threshold percentage of data.
    """
    print(f"Dropping columns missing more than {threshold*100}% of data...")
    
    # Calculate missing percentage per column
    missing_pct = df.isnull().sum() / len(df)
    
    # Identify columns to drop (e.g., EtCO2 which was 100% missing)
    columns_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    print(f"Removing columns: {columns_to_drop}")
    df = df.drop(columns=columns_to_drop)
    
    return df