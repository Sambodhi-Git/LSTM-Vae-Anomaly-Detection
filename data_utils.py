import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def generate_industrial_data(length=1000, noise_level=0.1):
    """
    Generates synthetic industrial sensor data.
    noise_level: 0.1 for Training (Clean), 0.5+ for Inference (Real-world)
    """
    t = np.linspace(0, 100, length)
    
    # 1. Temperature (deg C): Slow sine wave + trend
    temp = 80 + 10 * np.sin(t / 5) 
    
    # 2. Pressure (PSI): Constant base + Fast noise
    pressure = 100 + np.random.normal(0, 0.5, length)
    
    # 3. Torque (Nm): Faster sine wave
    torque = 50 + 20 * np.sin(t) 

    # Stack features
    data = np.stack([temp, pressure, torque], axis=1)
    
    # Add Sensor Noise
    data += np.random.normal(0, noise_level, data.shape)
    
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