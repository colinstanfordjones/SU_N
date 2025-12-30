import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x

class DenoisingNetwork(nn.Module):
    """
    Residual MLP for denoising 1D latent vectors.
    Inputs:
        x: (Batch, latent_dim) noisy latent vector
        t: (Batch,) time step
    """
    def __init__(self, latent_dim=1024, hidden_dim=1024, num_layers=6, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = x + t_emb # Add time embedding to features
            x = block(x)
            
        x = self.output_norm(x)
        return self.output_proj(x)

class NoiseScheduler:
    """
    Manages noise addition (forward) and sampling (reverse).
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define betas (linear schedule)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        
        # Pre-calculate diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_start, t):
        """
        Forward diffusion: q(x_t | x_0)
        Returns: x_t, noise
        """
        noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def sample(self, model, shape):
        """
        Reverse diffusion: p(x_{t-1} | x_t)
        Full loop sampling from T -> 0
        """
        model.eval()
        with torch.no_grad():
            img = torch.randn(shape, device=self.device)
            
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
                img = self.p_sample(model, img, t, i)
                
            return img

    def p_sample(self, model, x, t, t_index):
        """
        Single step of reverse diffusion.
        """
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

if __name__ == "__main__":
    # Test
    net = DenoisingNetwork(latent_dim=1024)
    x = torch.randn(4, 1024)
    t = torch.randint(0, 1000, (4,))
    out = net(x, t)
    print(f"Network output shape: {out.shape}")
    
    sched = NoiseScheduler()
    noisy, eps = sched.add_noise(x, t)
    print(f"Noisy shape: {noisy.shape}")
