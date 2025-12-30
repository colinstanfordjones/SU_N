import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    4D Spacetime Variational Autoencoder.
    Reshapes the 4th dimension into the channel dimension to use 3D convolutions.
    """
    def __init__(self, n_field, latent_dim=1024):
        super().__init__()
        self.n_field = n_field
        self.latent_dim = latent_dim
        
        # Channels per 3D slice: N_field * 2 (complex) * 16 (time slices)
        self.in_channels = n_field * 2 * 16
        
        # Encoder
        self.enc_conv1 = nn.Conv3d(self.in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.fc_mu = nn.Linear(256 * 2 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2 * 2, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256 * 2 * 2 * 2)
        self.dec_conv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose3d(64, self.in_channels, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        # x: (B, C, T, X, Y, Z) -> (B, C*T, X, Y, Z)
        B, C, T, X, Y, Z = x.shape
        x = x.view(B, C * T, X, Y, Z)
        
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = torch.flatten(h, start_dim=1)
        
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        h = h.view(-1, 256, 2, 2, 2)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        out = self.dec_conv3(h)
        
        # Reshape back to 4D
        B = out.shape[0]
        out = out.view(B, self.n_field * 2, 16, 16, 16, 16)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

if __name__ == "__main__":
    # Test with dummy data
    n_field = 1
    model = VAE(n_field=n_field)
    x = torch.randn(2, n_field * 2, 16, 16, 16, 16)
    recon, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Recon shape: {recon.shape}")
    print(f"Latent shape: {mu.shape}")
