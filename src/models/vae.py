import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import combinations


# ─── VAE Module ─────────────────────────────────────────────────────────────────
class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


# ─── Mutual Information Approximation ───────────────────────────────────────────
def mi_loss_gaussian(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def mutual_info_between_latents(z1, z2):
    n1, n2 = z1.shape[0], z2.shape[0]
    min_len = min(n1, n2)

    # idx1 = torch.randperm(n1)[:min_len]
    # idx2 = torch.randperm(n2)[:min_len]

    # z1_sample = z1[idx1]
    # z2_sample = z2[idx2]
    
    z1_sample = z1[:min_len]
    z2_sample = z2[:min_len]

    cos_sim = F.cosine_similarity(z1_sample, z2_sample, dim=1)
    return cos_sim.mean()

# z1~zn 서로 다르게  ! 
mi_min = False 

from tqdm import tqdm
# ─── 학습 준비 ─────────────────────────────────────────────────────────────────
# features_list: N개 그래프의 feature matrix list (torch.Tensor)
def train_vaes(features_list, hidden_dim=128, k=50, beta=1e-3, gamma=1e-2,  epochs=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vaes = []
    optims = []

    for feat in features_list:
        vae = GraphVAE(feat.shape[1], hidden_dim, k).to(device)
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
        vaes.append(vae)
        optims.append(opt)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        total_loss = 0.0
        zs = []

        for i, (x, vae, opt) in enumerate(zip(features_list, vaes, optims)):
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar, z = vae(x)

            # VAE loss (recon + KL)
            recon_loss = F.mse_loss(recon, x)
            kl_loss = mi_loss_gaussian(mu, logvar)

            # Self Mutual Info (approx as cosine sim)
            #info_recon = F.cosine_similarity(x, recon, dim=1).mean()

            # 기본 loss
            loss = recon_loss  + beta * kl_loss #- info_recon
            loss.backward()
            opt.step()

            total_loss += loss.item()
            zs.append(z.detach())

        # Graph-wise latent MI (cross-domain MI)
        mi_latent_loss = 0
        for z1, z2 in combinations(zs, 2):
            if mi_min: 
                mi_latent_loss += mutual_info_between_latents(z1, z2)
            else: 
                mi_latent_loss += -mutual_info_between_latents(z1, z2)

        

        # Optional: optimize cross-graph MI (joint step or separate)
        total_loss += gamma * mi_latent_loss
        tqdm.write(f"[Epoch {epoch}] Total Loss: {total_loss:.4f}, mi_latent_loss: {mi_latent_loss:.4f}")

    print('zs len: ', len(zs))
    print('zs[0]: ', zs[0].size())
    print(z[0][:50])
    return vaes, zs

def train_single_vae(x, hidden_dim=128, k=50, beta=1e-3, epochs=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.to(device)

    vae = GraphVAE(x.shape[1], hidden_dim, k).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in tqdm(range(epochs), desc="Training VAE on Target Graph"):
        vae.train()
        opt.zero_grad()

        recon, mu, logvar, z = vae(x)

        # VAE loss (recon + KL)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = mi_loss_gaussian(mu, logvar)

        # 기본 loss
        loss = recon_loss  + beta * kl_loss 
        loss.backward()
        opt.step()

        tqdm.write(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    # Return final latent representation
    vae.eval()
    with torch.no_grad():
        _, _, _, z_final = vae(x)
    print(z_final[:50])
    return vae, z_final

def train_vaes_bce(features_list, hidden_dim=128, k=50, beta=1e-3, gamma=1e-2,  epochs=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vaes = []
    optims = []
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    for feat in features_list:
        vae = GraphVAE(feat.shape[1], hidden_dim, k).to(device)
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
        vaes.append(vae)
        optims.append(opt)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        total_loss = 0.0
        zs = []

        for i, (x, vae, opt) in enumerate(zip(features_list, vaes, optims)):
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar, z  = vae(x)
            

            # VAE loss (recon + KL)
            recon_loss = loss_fn(recon, x)
            kl_loss = mi_loss_gaussian(mu, logvar)

            # Self Mutual Info (approx as cosine sim)
            #info_recon = F.cosine_similarity(x, recon, dim=1).mean()

            # 기본 loss
            loss = recon_loss  + beta * kl_loss #- info_recon
            loss.backward()
            opt.step()

            total_loss += loss.item()
            

            zs.append(z.detach())
        
        # Graph-wise latent MI (cross-domain MI)
        mi_latent_loss = 0
        for z1, z2 in combinations(zs, 2):
            if mi_min: 
                mi_latent_loss += mutual_info_between_latents(z1, z2)
            else: 
                mi_latent_loss += -mutual_info_between_latents(z1, z2)

        
        # (옵션) z1 normalize → 안정성 향상
        zs = [F.normalize(z, dim=1) for z in zs]  # 또는 std 기반 normalize
        # Optional: optimize cross-graph MI (joint step or separate)
        # total_loss += gamma * mi_latent_loss
        tqdm.write(f"[Epoch {epoch}] Total Loss: {total_loss:.4f}, mi_latent_loss: {mi_latent_loss:.4f}")

    print('zs len: ', len(zs))
    print('zs[0]: ', zs[0].size())
    print(z[0][:50])
    return vaes, zs

def train_single_vae_bce(x, hidden_dim=128, k=50, beta=1e-3, epochs=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.to(device)

    vae = GraphVAE(x.shape[1], hidden_dim, k).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    for epoch in tqdm(range(epochs), desc="Training VAE on Target Graph"):
        vae.train()
        opt.zero_grad()

        recon, mu, logvar, z = vae(x)
        print(recon.min().item(), recon.max().item())
        print(torch.isnan(recon).any())
        # VAE loss (recon + KL)
        recon_loss = loss_fn(recon, x)
        kl_loss = mi_loss_gaussian(mu, logvar)

        # 기본 loss
        loss = recon_loss  + beta * kl_loss 
        loss.backward()
        opt.step()

        tqdm.write(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    # Return final latent representation
    vae.eval()
    with torch.no_grad():
        _, _, _, z_final = vae(x)

    return vae, z_final
