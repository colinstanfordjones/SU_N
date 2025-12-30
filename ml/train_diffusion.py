import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SunDataset
from model import VAE
from diffusion import DenoisingNetwork, NoiseScheduler
import os
import argparse
from tqdm import tqdm

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    dataset = SunDataset(args.data_dir)
    if len(dataset) == 0:
        print(f"No samples found in {args.data_dir}")
        return
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Load Frozen VAE
    sample = dataset[0]
    n_field = sample.shape[0] // 2
    vae = VAE(n_field=n_field, latent_dim=args.latent_dim).to(device)
    
    if os.path.exists(args.vae_path):
        vae.load_state_dict(torch.load(args.vae_path, map_location=device))
        print(f"Loaded VAE from {args.vae_path}")
    else:
        print(f"Error: VAE model not found at {args.vae_path}. Cannot train diffusion.")
        return
        
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # 3. Initialize Diffusion Model
    model = DenoisingNetwork(latent_dim=args.latent_dim).to(device)
    scheduler = NoiseScheduler(num_timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            batch = batch.to(device)
            
            # Encode to latent space
            with torch.no_grad():
                mu, _ = vae.encode(batch)
                # We use the mean 'mu' as the target for diffusion
                # Optionally could sample from VAE posterior but mu is standard for LDM
                latents = mu.detach()

            # Sample timesteps
            t = torch.randint(0, args.timesteps, (latents.shape[0],), device=device).long()
            
            # Add noise
            noisy_latents, noise = scheduler.add_noise(latents, t)
            
            # Predict noise
            noise_pred = model(noisy_latents, t)
            
            # Loss
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"diffusion_epoch_{epoch+1}.pt"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "diffusion_final.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Latent Diffusion on su_n physics blocks")
    parser.add_argument("--data-dir", type=str, default="data/train", help="Directory with .sun files")
    parser.add_argument("--vae-path", type=str, default="ml/checkpoints/vae_final.pt", help="Path to pre-trained VAE")
    parser.add_argument("--checkpoint-dir", type=str, default="ml/checkpoints", help="Directory to save diffusion models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=1024, help="Latent dimension")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--save-interval", type=int, default=10, help="Save model every N epochs")
    
    args = parser.parse_args()
    train(args)
