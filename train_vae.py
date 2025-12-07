import sys, os
sys.path.insert(0, 'model_training')
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("\n" + "="*70)
print("TRAINING VAE")
print("="*70)

spark = SparkSession.builder.appName("Train-VAE").config("spark.driver.memory", "4g").getOrCreate()
df = spark.read.parquet('data/processed/ml_ready_data.parquet')
benign_df = df.filter(col('is_malicious') == 0)
print(f"Benign samples: {benign_df.count():,}")

pandas_df = benign_df.select('features').limit(10000).toPandas()
X_benign = np.array([row.toArray() for row in pandas_df['features']])
input_dim = X_benign.shape[1]
print(f"Training shape: {X_benign.shape}, Input dim: {input_dim}")

class SimpleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

model = SimpleVAE(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
X_tensor = torch.FloatTensor(X_benign)
train_loader = DataLoader(TensorDataset(X_tensor), batch_size=256, shuffle=True)

print("Training VAE (50 epochs)...")
model.train()
for epoch in range(50):
    total_loss = 0
    for (data,) in train_loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        recon_loss = nn.MSELoss()(recon, data)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.001 * kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    recon, _, _ = model(X_tensor[:1000])
    errors = torch.mean((X_tensor[:1000] - recon) ** 2, dim=1)
    threshold = torch.quantile(errors, 0.95).item()

os.makedirs('models/vae_binary', exist_ok=True)
torch.save({'model_state_dict': model.state_dict(), 'input_dim': input_dim, 'latent_dim': 8, 'threshold': threshold}, 'models/vae_binary/vae_model.pth')
spark.stop()
print("\n✅ VAE TRAINING COMPLETE!")
print(f"Model saved to: models/vae_binary/vae_model.pth")
