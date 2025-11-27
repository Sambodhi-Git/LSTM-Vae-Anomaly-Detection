import torch
import torch.nn as nn

class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len, device='cpu'):
        super(LSTM_VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.device = device

        # --- Encoder ---
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder ---
        # We project z back to hidden dimension to initialize decoder state
        self.fc_z_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # x shape: [batch, seq_len, input_dim]
        _, (h_n, _) = self.encoder_lstm(x)
        # h_n shape: [1, batch, hidden_dim] -> squeeze to [batch, hidden_dim]
        h_n = h_n.squeeze(0)
        
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z shape: [batch, latent_dim]
        
        # Create a sequence of z repeated seq_len times to feed the LSTM
        # Usually, we can also use z to init hidden state and feed zeros or the sequence.
        # Here we map z -> hidden and repeat it as input for simplicity in reconstruction
        
        batch_size = z.size(0)
        
        # Project z to hidden size
        hidden_map = self.fc_z_to_hidden(z) # [batch, hidden]
        
        # Expand to create a sequence input for decoder
        # We use the latent vector as input for every time step
        decoder_input = hidden_map.unsqueeze(1).repeat(1, self.seq_len, 1) # [batch, seq, hidden]
        
        out, _ = self.decoder_lstm(decoder_input) # [batch, seq, hidden]
        
        # Map back to feature space
        reconstruction = self.fc_output(out) # [batch, seq, input_dim]
        return reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar