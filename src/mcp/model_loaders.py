"""
Load all trained models for multi-stage pipeline
"""
from pyspark.ml.classification import GBTClassificationModel
import torch
import torch.nn as nn
import pickle
from pathlib import Path

class SimpleVAE(nn.Module):
    """VAE for anomaly detection"""
    def __init__(self, input_dim, latent_dim=8):
        super(SimpleVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
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
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class MalwareClassifier(nn.Module):
    """Neural Network for malware classification"""
    def __init__(self, input_dim, num_classes):
        super(MalwareClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class MultiModelLoader:
    """Loads all models for multi-stage pipeline"""
    
    def __init__(self, base_path="/Users/nidhirajani/Desktop/DIC Phase 3/models"):
        self.base_path = Path(base_path)
        self.models = {}
        
    def load_all(self):
        """Load all models"""
        print("🤖 Loading all models...")
        
        # Load LightGBM
        try:
            lgbm_path = self.base_path / "lightgbm_classifier"
            self.models['lightgbm'] = GBTClassificationModel.load(str(lgbm_path))
            print("  ✅ LightGBM loaded")
        except Exception as e:
            print(f"  ❌ LightGBM failed: {e}")
        
        # Load VAE
        try:
            vae_path = self.base_path / "vae_binary" / "vae_model.pth"
            checkpoint = torch.load(vae_path)
            vae_model = SimpleVAE(checkpoint['input_dim'], checkpoint['latent_dim'])
            vae_model.load_state_dict(checkpoint['model_state_dict'])
            vae_model.eval()
            self.models['vae'] = {
                'model': vae_model,
                'threshold': checkpoint['threshold']
            }
            print("  ✅ VAE loaded")
        except Exception as e:
            print(f"  ❌ VAE failed: {e}")
        
        # Load Neural Network
        try:
            nn_path = self.base_path / "neural_net_multiclass" / "nn_model.pth"
            checkpoint = torch.load(nn_path)
            nn_model = MalwareClassifier(checkpoint['input_dim'], checkpoint['num_classes'])
            nn_model.load_state_dict(checkpoint['model_state_dict'])
            nn_model.eval()
            self.models['neural_net'] = {
                'model': nn_model,
                'label_encoder': checkpoint['label_encoder']
            }
            print("  ✅ Neural Network loaded")
        except Exception as e:
            print(f"  ❌ Neural Network failed: {e}")
        
        print(f"✅ Loaded {len(self.models)} models\n")
        return self.models
    
    def get_model(self, name):
        """Get specific model"""
        return self.models.get(name)
