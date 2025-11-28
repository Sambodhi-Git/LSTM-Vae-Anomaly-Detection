import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import joblib 
from sklearn.preprocessing import MinMaxScaler

from lstm_vae import LSTM_VAE
from data_utils import generate_industrial_data, save_to_excel, create_sliding_windows, TimeSeriesDataset

def train():
    SEQ_LEN = 30
    BATCH_SIZE = 32
    EPOCHS = 60
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- KEY CHANGE 1: Train on CLEAN Data ---
    print("Generating CLEAN Training Data...")
    df = generate_industrial_data(length=2000, noise_level=0.1) # Low noise
    save_to_excel(df, 'training_data.xlsx')

    # Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)
    joblib.dump(scaler, 'scaler.pkl')

    # Windows
    windows = create_sliding_windows(data_scaled, SEQ_LEN)
    dataset = TimeSeriesDataset(windows)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = LSTM_VAE(input_dim=3, hidden_dim=64, latent_dim=16, seq_len=SEQ_LEN, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss(reduction='sum')

    model.train()
    print("Training Model...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = criterion(recon, batch) # Simple MSE is enough for this
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss {total_loss/len(dataset):.4f}")

    torch.save(model.state_dict(), 'lstm_vae_model_industrial.pth')
    print("Model saved.")

if __name__ == "__main__":
    train()