import torch
import numpy as np
from model import VAE
from diffusion import DenoisingNetwork, NoiseScheduler
import argparse
import struct
import os

def save_sun_file(tensor, path, n_field, block_size=16):
    """
    Save a generated tensor to a .sun file.
    tensor: (N_field*2, 16, 16, 16, 16)
    """
    # Reshape back to interleave format: (16, 16, 16, 16, N_field, 2)
    # Input is (C, T, X, Y, Z)
    # C = N_field * 2
    
    # (C, T, X, Y, Z) -> (T, X, Y, Z, C)
    tensor = tensor.permute(1, 2, 3, 4, 0)
    
    # (T, X, Y, Z, C) -> (T, X, Y, Z, N_field, 2)
    tensor = tensor.reshape(block_size, block_size, block_size, block_size, n_field, 2)
    
    # Convert to numpy and flatten
    data = tensor.detach().cpu().numpy().astype('<f8').tobytes()
    
    # Create header
    # magic: [4]u8 = "SUN\0"
    # version: u32 = 1
    # num_blocks: u32 = 1
    # block_size: u32
    # n_gauge: u32 = 1 (default)
    # n_field: u32
    # base_spacing: f64 = 1.0 (dummy)
    # time_slice: u32 = 0xFFFFFFFF
    # flags: u32 = 8 (FLAG_HAS_FULL_PSI)
    
    header = struct.pack('<4sIIIIIdII', b'SUN\x00', 1, 1, block_size, 1, n_field, 1.0, 0xFFFFFFFF, 8)
    
    # Block Header
    # level: u8 = 0
    # padding: [7]u8
    # origin: [4]u64 = 0
    block_header = struct.pack('<B7sQQQQ', 0, b'\x00'*7, 0, 0, 0, 0)
    
    with open(path, 'wb') as f:
        f.write(header)
        f.write(block_header)
        f.write(data)

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load VAE
    vae = VAE(n_field=args.n_field, latent_dim=args.latent_dim).to(device)
    if os.path.exists(args.vae_path):
        vae.load_state_dict(torch.load(args.vae_path, map_location=device))
    else:
        print("VAE model not found.")
        return
    vae.eval()
    
    # Load Diffusion
    model = DenoisingNetwork(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    if os.path.exists(args.diffusion_path):
        model.load_state_dict(torch.load(args.diffusion_path, map_location=device))
    else:
        print("Diffusion model not found.")
        return
    model.eval()
    
    scheduler = NoiseScheduler(num_timesteps=args.timesteps, device=device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_samples} samples...")
    
    # Sampling Loop
    # 1. Generate latent vector from noise
    latents = scheduler.sample(model, (args.num_samples, args.latent_dim))
    
    # 2. Decode latents to physics blocks
    with torch.no_grad():
        generated_blocks = vae.decode(latents)
        
    # 3. Save to disk
    for i in range(args.num_samples):
        path = os.path.join(args.output_dir, f"gen_{i:04d}.sun")
        save_sun_file(generated_blocks[i], path, args.n_field)
        print(f"Saved {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", type=str, default="ml/checkpoints/vae_final.pt")
    parser.add_argument("--diffusion-path", type=str, default="ml/checkpoints/diffusion_final.pt")
    parser.add_argument("--output-dir", type=str, default="data/generated")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--n-field", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=1000)
    
    args = parser.parse_args()
    generate(args)
