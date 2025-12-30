import torch
from dataset import SunDataset
from model import VAE
from vector_db import SimpleVectorDB
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

def encode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = SunDataset(args.data_dir)
    if len(dataset) == 0:
        print("Dataset is empty.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Detect n_field
    sample = dataset[0]
    n_field = sample.shape[0] // 2
    
    model = VAE(n_field=n_field, latent_dim=args.latent_dim).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model not found at {args.model_path}. Using uninitialized model (random embeddings).")
        
    model.eval()
    
    db = SimpleVectorDB(latent_dim=args.latent_dim)
    
    print("Encoding dataset...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            if data is None: continue
            mu, _ = model.encode(data.to(device))
            
            for j in range(len(data)):
                global_idx = i * args.batch_size + j
                if global_idx >= len(dataset.block_data): break
                
                meta = dataset.block_data[global_idx]
                # Filter out unnecessary meta for the DB file size
                compact_meta = {
                    "file": os.path.basename(meta["file"]),
                    "offset": meta["offset"],
                    "level": meta["level"] if "level" in meta else 0,
                    "origin": meta["origin"] if "origin" in meta else [0,0,0,0]
                }
                db.add(mu[j], compact_meta)
                
    db.build_index()
    db.save(args.db_path)
    print(f"Database saved to {args.db_path} with {len(db.vectors)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/train", help="Directory with .sun files")
    parser.add_argument("--model-path", type=str, default="ml/checkpoints/vae_final.pt", help="Path to trained VAE weights")
    parser.add_argument("--db-path", type=str, default="ml/physics_db.npz", help="Path to save the vector DB")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding")
    parser.add_argument("--latent-dim", type=int, default=1024, help="Latent dimension")
    args = parser.parse_args()
    encode(args)
