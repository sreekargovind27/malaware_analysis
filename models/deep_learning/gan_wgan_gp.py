"""
WGAN-GP for Benign Manifold Modeling + Anomaly Scoring (Semi-Supervised Version)

This trains a Wasserstein GAN with Gradient Penalty (WGAN-GP)
on "core benign" flows (already cleaned by IsolationForest in get_data_for_autoencoder()),
with optional semi-supervised training using malicious samples.

The Generator learns to produce realistic benign-like samples and
the Critic learns to score how 'benign' a sample looks, while also
learning to recognize malicious samples as "fake".

We:
  - run Optuna to tune generator/critic depth/width/z_dim/lr/etc.
  - fully train the best model with patience
  - optionally inject malicious samples during critic training
  - save generator, critic, scaler, feature_list, and hyperparams to disk
"""

import os

import matplotlib.pyplot as plt
import optuna
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import Config
from models.data_loader import get_data_for_autoencoder

# -------------------------------------------------
# Fallback: WGAN-GP needs 2nd-order grads (gradient penalty),
# which MPS can't do in torch==2.0.1. Run on CPU instead.
# -------------------------------------------------
if getattr(Config, "DEVICE", None) == "mps":
    print("⚠ MPS can't handle 2nd-order grads for WGAN-GP. Falling back to CPU.")
    Config.DEVICE = "cpu"


# -------------------------------------------------

# -------------------------
# Load malicious samples for semi-supervised training
# -------------------------
def get_malicious_samples(ratio=0.1):
    """
    Load a small subset of malicious samples for semi-supervised training.

    Args:
        ratio: Fraction of malicious samples to use (e.g., 0.1 = 10%)

    Returns:
        malicious_np: Malicious samples as numpy array (unscaled)
    """
    import pandas as pd
    import joblib

    print(f"\n📊 Loading {ratio * 100:.0f}% malicious samples for semi-supervised training...")

    # Load full dataset
    df = pd.read_parquet(Config.ENGINEERED_DATA_PATH)

    # Load feature list (same as autoencoder uses)
    feature_list = joblib.load(Config.AUTOENCODER_FEATURE_LIST_PATH)

    # Get malicious samples
    malicious_df = df[df[Config.TARGET_COL] == 'Malicious'][feature_list].fillna(0)

    # Sample a subset
    n_samples = int(len(malicious_df) * ratio)
    malicious_subset = malicious_df.sample(
        n=min(n_samples, len(malicious_df)),
        random_state=Config.RANDOM_STATE
    )

    print(f"   Selected {len(malicious_subset):,} malicious samples")

    return malicious_subset.values.astype('float32')


# -------------------------
# Utility: simple MLP builder
# -------------------------
def build_mlp(in_dim, hidden_dims, out_dim, dropout_rate=0.0, final_activation=None):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LeakyReLU(0.2))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


# -------------------------
# Generator
# -------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dims, out_dim, dropout_rate=0.0):
        super().__init__()
        # last layer: no activation, we'll output raw feature space
        self.net = build_mlp(
            in_dim=z_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            dropout_rate=dropout_rate,
            final_activation=None,
        )

    def forward(self, z):
        return self.net(z)


# -------------------------
# Critic (WGAN critic, NOT sigmoid)
# -------------------------
class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout_rate=0.0):
        super().__init__()
        # final layer -> 1 scalar, no activation
        self.net = build_mlp(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=1,
            dropout_rate=dropout_rate,
            final_activation=None,
        )

    def forward(self, x):
        return self.net(x).view(-1)


# -------------------------
# Gradient penalty for WGAN-GP
# -------------------------
def gradient_penalty(critic, real_samples, fake_samples, device, lambda_gp=10.0):
    batch_size = real_samples.size(0)

    eps = torch.rand(batch_size, 1, device=device)
    eps = eps.expand_as(real_samples)

    # interpolate
    interpolates = eps * real_samples + (1 - eps) * fake_samples
    interpolates.requires_grad_(True)

    pred = critic(interpolates)

    grads = autograd.grad(
        outputs=pred,
        inputs=interpolates,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean() * lambda_gp
    return gp


# -------------------------
# Score function for validation during Optuna
# We estimate Wasserstein distance on val set:
#   W = mean(D(real_val)) - mean(D(fake_val))
# Larger W is better => lower loss = -W
# -------------------------
def evaluate_wasserstein(generator, critic, val_tensor, z_dim, device, n_fake=2048):
    critic.eval()
    generator.eval()

    with torch.no_grad():
        # sample a batch of real
        if val_tensor.size(0) > n_fake:
            idx = torch.randint(0, val_tensor.size(0), (n_fake,), device=device)
            real_batch = val_tensor[idx]
        else:
            real_batch = val_tensor

        # sample fake
        z = torch.randn(real_batch.size(0), z_dim, device=device)
        fake_batch = generator(z)

        real_score = critic(real_batch).mean()
        fake_score = critic(fake_batch).mean()

    W = real_score.item() - fake_score.item()
    return W


# -------------------------
# Trainer class
# -------------------------
class WGAN_GP_Model:
    def __init__(self, input_dim):
        self.device = Config.DEVICE
        self.input_dim = input_dim

        # populated later (Optuna or best params)
        self.gen_hidden = [128, 64]
        self.dis_hidden = [128, 64]
        self.z_dim = 32
        self.dropout_rate = 0.1
        self.g_lr = 1e-4
        self.d_lr = 1e-4
        self.n_critic = 5
        self.noise_factor = 0.0  # not used like AE, but keeping symmetry
        self.best_params = None

        # Semi-supervised parameters
        self.use_malicious = getattr(Config, "GAN_USE_MALICIOUS", True)
        self.malicious_ratio = getattr(Config, "GAN_MALICIOUS_RATIO", 0.1)
        self.malicious_weight = getattr(Config, "GAN_MALICIOUS_WEIGHT", 0.1)

        # saved artifacts
        self.scaler = None
        self.feature_list = None  # <- IMPORTANT: exact column order used for training

        # built later
        self.generator = None
        self.critic = None
        self.opt_g = None
        self.opt_d = None

        print(f"WGAN-GP initialized on {self.device}")
        if self.use_malicious:
            print(f"  Semi-supervised mode enabled:")
            print(f"    Malicious ratio: {self.malicious_ratio}")
            print(f"    Malicious weight: {self.malicious_weight}")

    def build_models(self):
        self.generator = Generator(
            z_dim=self.z_dim,
            hidden_dims=self.gen_hidden,
            out_dim=self.input_dim,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        self.critic = Critic(
            in_dim=self.input_dim,
            hidden_dims=self.dis_hidden,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.g_lr,
            betas=(0.5, 0.9),
        )
        self.opt_d = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.d_lr,
            betas=(0.5, 0.9),
        )

    def save_model(self, filename='gan_wgan_gp_final.pth', threshold=None):
        os.makedirs(Config.DL_MODELS_DIR, exist_ok=True)
        filepath = os.path.join(Config.DL_MODELS_DIR, filename)
        torch.save(
            {
                'generator_state': self.generator.state_dict(),
                'critic_state': self.critic.state_dict(),
                'scaler': self.scaler,
                'input_dim': self.input_dim,
                'z_dim': self.z_dim,
                'gen_hidden': self.gen_hidden,
                'dis_hidden': self.dis_hidden,
                'dropout_rate': self.dropout_rate,
                'g_lr': self.g_lr,
                'd_lr': self.d_lr,
                'n_critic': self.n_critic,
                'best_params': self.best_params,
                'threshold': threshold,
                'final_feature_list': self.feature_list,  # <- SAVE FEATURE ORDER
                'use_malicious': self.use_malicious,
                'malicious_ratio': self.malicious_ratio,
                'malicious_weight': self.malicious_weight,
            },
            filepath,
        )

    @classmethod
    def load_from_checkpoint(cls, filename='gan_wgan_gp_final.pth'):
        filepath = os.path.join(Config.DL_MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(
            filepath,
            map_location=Config.DEVICE,
            weights_only=False,
        )

        # Recreate instance
        instance = cls(input_dim=checkpoint['input_dim'])

        # restore hyperparams
        instance.z_dim = checkpoint['z_dim']
        instance.gen_hidden = checkpoint['gen_hidden']
        instance.dis_hidden = checkpoint['dis_hidden']
        instance.dropout_rate = checkpoint['dropout_rate']
        instance.g_lr = checkpoint['g_lr']
        instance.d_lr = checkpoint['d_lr']
        instance.n_critic = checkpoint['n_critic']
        instance.best_params = checkpoint.get('best_params', None)
        instance.scaler = checkpoint['scaler']
        instance.feature_list = checkpoint['final_feature_list']

        # Semi-supervised params
        instance.use_malicious = checkpoint.get('use_malicious', False)
        instance.malicious_ratio = checkpoint.get('malicious_ratio', 0.1)
        instance.malicious_weight = checkpoint.get('malicious_weight', 0.1)

        # rebuild
        instance.build_models()
        instance.generator.load_state_dict(checkpoint['generator_state'])
        instance.critic.load_state_dict(checkpoint['critic_state'])
        instance.generator.eval()
        instance.critic.eval()

        print(f"✅ WGAN-GP loaded from: {filename}")

        # build a basic threshold for anomaly scoring if not present
        threshold_val = checkpoint.get('threshold', None)
        return instance, threshold_val

    # -------------------------
    # Training epoch with semi-supervised malicious injection
    # -------------------------
    def _train_epoch(self, train_tensor, batch_size, lambda_gp, max_critic_steps,
                     malicious_tensor=None, malicious_weight=0.1):
        """
        One training epoch for WGAN-GP with optional semi-supervised malicious samples.

        Args:
            train_tensor: Benign training data
            batch_size: Batch size
            lambda_gp: Gradient penalty weight
            max_critic_steps: Critic updates per generator update
            malicious_tensor: Optional malicious samples (treated as "fake")
            malicious_weight: Probability of using malicious samples per batch
        """
        self.generator.train()
        self.critic.train()

        n = train_tensor.size(0)
        perm = torch.randperm(n, device=self.device)

        g_loss_running = 0.0
        d_loss_running = 0.0
        steps = 0

        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            if end_idx - start_idx < 2:
                continue

            bs = end_idx - start_idx
            idx = perm[start_idx:end_idx]
            real_batch = train_tensor[idx]

            # --- Train Critic multiple times
            for _ in range(max_critic_steps):
                self.opt_d.zero_grad()

                # Real samples (benign)
                d_real = self.critic(real_batch).mean()

                # Fake samples: mix of generator output + malicious (if available)
                z = torch.randn(bs, self.z_dim, device=self.device)
                gen_samples = self.generator(z).detach()

                # Semi-supervised: occasionally replace generated samples with malicious
                if malicious_tensor is not None and torch.rand(1).item() < malicious_weight:
                    # Sample from malicious data
                    mal_idx = torch.randint(0, malicious_tensor.size(0), (bs,), device=self.device)
                    fake_batch = malicious_tensor[mal_idx]
                else:
                    # Use generator samples
                    fake_batch = gen_samples

                d_fake = self.critic(fake_batch).mean()
                wass_loss = -(d_real - d_fake)  # want to maximize (d_real - d_fake)

                gp = gradient_penalty(
                    self.critic,
                    real_batch,
                    fake_batch,
                    self.device,
                    lambda_gp=lambda_gp,
                )
                d_total = wass_loss + gp
                d_total.backward()
                self.opt_d.step()

            # --- Train Generator once
            z = torch.randn(bs, self.z_dim, device=self.device)
            gen_samples = self.generator(z)
            self.opt_g.zero_grad()
            g_loss = -self.critic(gen_samples).mean()  # maximize critic(fake)
            g_loss.backward()
            self.opt_g.step()

            g_loss_running += g_loss.item()
            d_loss_running += d_total.item()
            steps += 1

        avg_g = g_loss_running / max(steps, 1)
        avg_d = d_loss_running / max(steps, 1)
        return avg_g, avg_d

    # -------------------------
    # validation metric during full training
    # -------------------------
    def _val_wasserstein(self, val_tensor):
        with torch.no_grad():
            self.generator.eval()
            self.critic.eval()

            # sample ~2048 pairs
            n_fake = min(2048, val_tensor.size(0))
            idx = torch.randint(0, val_tensor.size(0), (n_fake,), device=self.device)
            val_real = val_tensor[idx]
            z = torch.randn(n_fake, self.z_dim, device=self.device)
            val_fake = self.generator(z)

            d_real = self.critic(val_real).mean().item()
            d_fake = self.critic(val_fake).mean().item()
            W = d_real - d_fake
        return W

    # -------------------------
    # Optuna objective
    # -------------------------
    def _optuna_objective(self, trial, train_tensor, val_tensor, malicious_tensor=None):
        # Sample hyperparams
        self.z_dim = trial.suggest_int('z_dim', 8, 64)

        h1 = trial.suggest_int('gen_hidden_1', 32, 256)
        h2 = trial.suggest_int('gen_hidden_2', 16, 128)
        self.gen_hidden = [h1, h2]

        d1 = trial.suggest_int('dis_hidden_1', 32, 256)
        d2 = trial.suggest_int('dis_hidden_2', 16, 128)
        self.dis_hidden = [d1, d2]

        self.dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.4)
        self.g_lr = trial.suggest_float('g_lr', 1e-5, 5e-3, log=True)
        self.d_lr = trial.suggest_float('d_lr', 1e-5, 5e-3, log=True)
        self.n_critic = trial.suggest_int('n_critic', 3, 7)

        # build models fresh with sampled params
        self.build_models()

        # quick train loop (fewer epochs than final)
        quick_epochs = min(10, Config.AUTOENCODER_EPOCHS)
        for _ in range(quick_epochs):
            self._train_epoch(
                train_tensor=train_tensor,
                batch_size=Config.AUTOENCODER_BATCH_SIZE,
                lambda_gp=10.0,
                max_critic_steps=self.n_critic,
                malicious_tensor=malicious_tensor if self.use_malicious else None,
                malicious_weight=self.malicious_weight,
            )

        # evaluate on val
        W = evaluate_wasserstein(
            self.generator,
            self.critic,
            val_tensor,
            self.z_dim,
            self.device,
        )
        # bigger W is better => we minimize negative
        return -W

    def optimize_hyperparameters(self, train_tensor, val_tensor, malicious_tensor=None):
        print("\n" + "=" * 70)
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (WGAN-GP)")
        if self.use_malicious and malicious_tensor is not None:
            print(f"   Semi-supervised mode: Using {malicious_tensor.size(0)} malicious samples")
        print("=" * 70)

        if not Config.USE_OPTUNA:
            print("\n⭐ Optuna disabled in config, using default WGAN-GP params")
            self.best_params = {
                'z_dim': self.z_dim,
                'gen_hidden_1': self.gen_hidden[0],
                'gen_hidden_2': self.gen_hidden[1],
                'dis_hidden_1': self.dis_hidden[0],
                'dis_hidden_2': self.dis_hidden[1],
                'dropout_rate': self.dropout_rate,
                'g_lr': self.g_lr,
                'd_lr': self.d_lr,
                'n_critic': self.n_critic,
            }
            return self.best_params

        def objective_wrapper(trial):
            return self._optuna_objective(trial, train_tensor, val_tensor, malicious_tensor)

        study = optuna.create_study(direction='minimize')
        study.optimize(
            objective_wrapper,
            n_trials=Config.OPTUNA_N_TRIALS,
            timeout=Config.OPTUNA_TIMEOUT,
        )

        self.best_params = study.best_params

        # write best params back into self
        self.z_dim = self.best_params['z_dim']
        self.gen_hidden = [
            self.best_params['gen_hidden_1'],
            self.best_params['gen_hidden_2'],
        ]
        self.dis_hidden = [
            self.best_params['dis_hidden_1'],
            self.best_params['dis_hidden_2'],
        ]
        self.dropout_rate = self.best_params['dropout_rate']
        self.g_lr = self.best_params['g_lr']
        self.d_lr = self.best_params['d_lr']
        self.n_critic = self.best_params['n_critic']

        return self.best_params

    def train_full(self, train_tensor, val_tensor, scaler, feature_list, malicious_tensor=None):
        # store artifacts needed for eval
        self.scaler = scaler
        self.feature_list = feature_list

        # rebuild final models with chosen/best params
        print("\n🏗️  Building final WGAN-GP with:")
        print(f"    gen_hidden={self.gen_hidden}")
        print(f"    dis_hidden={self.dis_hidden}")
        print(f"    z_dim={self.z_dim}")
        print(f"    dropout_rate={self.dropout_rate}")
        print(f"    g_lr={self.g_lr}")
        print(f"    d_lr={self.d_lr}")
        print(f"    n_critic={self.n_critic}")
        if self.use_malicious and malicious_tensor is not None:
            print(f"    semi-supervised: {malicious_tensor.size(0)} malicious samples")
            print(f"    malicious_weight: {self.malicious_weight}")
        self.build_models()

        best_W = -1e9
        patience = 10
        patience_counter = 0

        g_hist = []
        d_hist = []
        w_hist = []

        for epoch in tqdm(range(Config.AUTOENCODER_EPOCHS), desc="Training WGAN-GP"):
            g_loss, d_loss = self._train_epoch(
                train_tensor=train_tensor,
                batch_size=Config.AUTOENCODER_BATCH_SIZE,
                lambda_gp=10.0,
                max_critic_steps=self.n_critic,
                malicious_tensor=malicious_tensor if self.use_malicious else None,
                malicious_weight=self.malicious_weight,
            )

            # validation W
            W_val = self._val_wasserstein(val_tensor)

            g_hist.append(g_loss)
            d_hist.append(d_loss)
            w_hist.append(W_val)

            if W_val > best_W:
                best_W = W_val
                patience_counter = 0
                # save best-so-far snapshot
                self.save_model('gan_wgan_gp_best.pth', threshold=None)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                    break

        # reload best
        best_model, _ = WGAN_GP_Model.load_from_checkpoint('gan_wgan_gp_best.pth')

        # copy best weights/artifacts into self
        self.generator.load_state_dict(best_model.generator.state_dict())
        self.critic.load_state_dict(best_model.critic.state_dict())
        self.scaler = best_model.scaler
        self.feature_list = best_model.feature_list

        # final save
        self.save_model('gan_wgan_gp_final.pth', threshold=None)

        # plot training curves
        self._plot_training_history(g_hist, d_hist, w_hist)

    def _plot_training_history(self, g_hist, d_hist, w_hist):
        os.makedirs(Config.GAN_RESULTS_DIR, exist_ok=True)

        # G / D loss
        plt.figure(figsize=(10, 6))
        plt.plot(g_hist, label='Generator Loss')
        plt.plot(d_hist, label='Critic Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('WGAN-GP Training Loss (Semi-Supervised)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.GAN_RESULTS_DIR, 'wgan_gp_training_losses.png'))
        plt.close()

        # Wasserstein est
        plt.figure(figsize=(10, 6))
        plt.plot(w_hist, label='Val Wasserstein Estimate (higher=better)')
        plt.xlabel('Epoch')
        plt.ylabel('W Distance')
        plt.title('WGAN-GP Validation Wasserstein Distance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.GAN_RESULTS_DIR, 'wgan_gp_val_wasserstein.png'))
        plt.close()


# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    Config.set_seeds()
    Config.ensure_output_dirs()

    print("\n" + "=" * 70)
    print("🚀 WGAN-GP TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Device: {Config.DEVICE}")
    print(f"Batch size: {Config.AUTOENCODER_BATCH_SIZE}")
    print(f"Optuna enabled: {Config.USE_OPTUNA}")
    print(f"Semi-supervised: {getattr(Config, 'GAN_USE_MALICIOUS', True)}")
    print(f"Epochs: {Config.AUTOENCODER_EPOCHS}")
    print("=" * 70)

    # Load benign data
    train_loader, val_loader, scaler, final_feature_list = get_data_for_autoencoder()
    if (not final_feature_list) or (len(train_loader.dataset) == 0):
        raise RuntimeError("No benign data available for GAN training.")

    input_dim = len(final_feature_list)


    def loader_to_tensor(dl):
        xs = []
        for batch in dl:
            if isinstance(batch, tuple):
                batch = batch[0]
            xs.append(batch)
        return torch.cat(xs, dim=0)


    train_tensor = loader_to_tensor(train_loader).to(Config.DEVICE)
    val_tensor = loader_to_tensor(val_loader).to(Config.DEVICE)

    # Load malicious samples
    malicious_tensor = None
    if getattr(Config, 'GAN_USE_MALICIOUS', True):
        malicious_np = get_malicious_samples(ratio=getattr(Config, 'GAN_MALICIOUS_RATIO', 0.1))
        malicious_scaled = scaler.transform(malicious_np)
        malicious_tensor = torch.FloatTensor(malicious_scaled).to(Config.DEVICE)
        print(f"✅ Loaded {malicious_tensor.size(0):,} malicious samples")

    # Check if we want to test multiple injection rates
    TEST_INJECTION_RATES = getattr(Config, 'GAN_TEST_INJECTION_RATES', False)

    if TEST_INJECTION_RATES:
        # GRID SEARCH MODE
        print("\n" + "=" * 70)
        print("🧪 INJECTION RATE GRID SEARCH MODE")
        print("=" * 70)

        # First run Optuna to find best architecture
        gan_model = WGAN_GP_Model(input_dim)
        best_params = gan_model.optimize_hyperparameters(train_tensor, val_tensor, malicious_tensor)

        # Test different injection rates
        test_rates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        results = {}

        for rate in test_rates:
            print(f"\n{'=' * 70}")
            print(f"Testing injection rate: {rate}")
            print("=" * 70)

            model = WGAN_GP_Model(input_dim)
            model.z_dim = best_params['z_dim']
            model.gen_hidden = [best_params['gen_hidden_1'], best_params['gen_hidden_2']]
            model.dis_hidden = [best_params['dis_hidden_1'], best_params['dis_hidden_2']]
            model.dropout_rate = best_params['dropout_rate']
            model.g_lr = best_params['g_lr']
            model.d_lr = best_params['d_lr']
            model.n_critic = best_params['n_critic']
            model.malicious_weight = rate

            model.train_full(train_tensor, val_tensor, scaler, final_feature_list, malicious_tensor)
            model.save_model(f'gan_wgan_gp_rate_{int(rate * 100)}.pth')

            # Store for comparison (you need to eval separately)
            print(f"✅ Completed rate {rate}")

        print("\n🏁 Grid search complete. Evaluate each model separately.")

    else:
        # NORMAL MODE - single run with config rate
        gan_model = WGAN_GP_Model(input_dim)
        gan_model.optimize_hyperparameters(train_tensor, val_tensor, malicious_tensor)
        gan_model.train_full(train_tensor, val_tensor, scaler, final_feature_list, malicious_tensor)
        print("✅ WGAN-GP training complete.")
