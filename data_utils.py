import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def generate_industrial_data(length=1000, noise_level=0.5):
    """
    Generates synthetic industrial sensor data.
    """
    t = np.linspace(0, 100, length)
    
    # 1. Temperature (deg C): Slow sine wave + Base 80
    # Normal range: ~70 to 90
    temp = 80 + 10 * np.sin(t / 5) 
    
    # 2. Pressure (PSI): Constant base 100 + High freq noise
    # Normal range: ~95 to 105
    pressure = 100 + np.random.normal(0, 1.5, length)
    
    # 3. Torque (Nm): Faster sine wave + Base 50
    # Normal range: ~30 to 70
    torque = 50 + 20 * np.sin(t) 

    # Combine
    data = np.stack([temp, pressure, torque], axis=1)
    
    # Add general sensor noise
    data += np.random.normal(0, noise_level, data.shape)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Temperature', 'Pressure', 'Torque'])
    
    return df

def save_to_excel(df, filename='sensor_data.xlsx'):
    df.to_excel(filename, index=False)
    print(f"Data saved to {filename}")

def create_sliding_windows(data, seq_length):
    x = []
    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length)]
        x.append(_x)
    return np.array(x)

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]).float()