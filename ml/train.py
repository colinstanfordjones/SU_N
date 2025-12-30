import torch
from torch.utils.data import DataLoader
from dataset import SunDataset
from model import VAE, vae_loss
import os
import argparse
from tqdm import tqdm

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SunDataset(args.data_dir)
    if len(dataset) == 0:
        print(f"No samples found in {args.data_dir}")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Assume n_field=1 for now, we should detect it from dataset
    # Actually, we can get it from the first sample
    n_field = dataset[0].shape[0] // 2
    model = VAE(n_field=n_field, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item() / len(data)})

        avg_loss = train_loss / len(dataset)
        print(f"====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"vae_epoch_{epoch+1}.pt"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "vae_final.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE on su_n physics blocks")
    parser. advertisement = "Train VAE on su_n physics blocks"
    parser.add_argument("--data-dir", type=str, default="data/train", help="Directory with .sun files")
    parser.add_argument("--checkpoint-dir", type=str, default="ml/checkpoints", help="Directory to save models")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=1024, help="Latent dimension")
    parser.add_argument("--save-interval", type=int, default=10, help="Save model every N epochs")
    
    args = parser.parse_args()
    train(args)
