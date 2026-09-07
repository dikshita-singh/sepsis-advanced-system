import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates advanced clinical signs to help the model detect early sepsis.
    """
    print("Engineering advanced clinical features...")
    
    # Avoid dividing by zero by adding a tiny value (eps)
    eps = 1e-5
    
    # 1. Calculate Shock Index (Heart Rate / Systolic Blood Pressure)
    if 'HR' in df.columns and 'SBP' in df.columns:
        df['shock_index'] = df['HR'] / (df['SBP'] + eps)
        
    # 2. Calculate Pulse Pressure (Systolic BP - Diastolic BP)
    if 'SBP' in df.columns and 'DBP' in df.columns:
        df['pulse_pressure'] = df['SBP'] - df['DBP']
        
    # 3. Clean up any extreme values or infinity markers created by divisions
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fill any newly created NaNs with 0
    df = df.fillna(0)
    
    print("Feature engineering complete.")
    return df