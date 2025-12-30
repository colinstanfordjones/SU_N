import unittest
import torch
import os
import shutil
import subprocess
from ml.model import VAE
from ml.diffusion import DenoisingNetwork, NoiseScheduler
from ml.dataset import SunDataset

class TestLatentDiffusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "tests/ml_diff_temp"
        cls.data_dir = os.path.join(cls.test_dir, "data")
        cls.ckpt_dir = os.path.join(cls.test_dir, "checkpoints")
        cls.gen_dir = os.path.join(cls.test_dir, "generated")
        
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.ckpt_dir, exist_ok=True)
        os.makedirs(cls.gen_dir, exist_ok=True)
        
        # 1. Generate Synthetic Data
        print("Generating synthetic dataset...")
        subprocess.run(
            ["zig", "build", "generate-dataset", "--", "5", cls.data_dir],
            check=True,
            capture_output=True
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_full_pipeline(self):
        device = torch.device("cpu")
        latent_dim = 64
        n_field = 1
        
        # 1. Train VAE (Mock training)
        vae = VAE(n_field=n_field, latent_dim=latent_dim).to(device)
        dataset = SunDataset(self.data_dir)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # One pass to init weights and save
        vae_path = os.path.join(self.ckpt_dir, "vae.pt")
        torch.save(vae.state_dict(), vae_path)
        
        # 2. Train Diffusion (Mock training)
        diff_model = DenoisingNetwork(latent_dim=latent_dim, hidden_dim=64, num_layers=2).to(device)
        optimizer = torch.optim.Adam(diff_model.parameters(), lr=1e-3)
        scheduler = NoiseScheduler(num_timesteps=10, device=device)
        
        diff_model.train()
        for batch in dataloader:
            batch = batch.to(device)
            with torch.no_grad():
                mu, _ = vae.encode(batch)
            
            t = torch.randint(0, 10, (mu.shape[0],), device=device)
            noisy, noise = scheduler.add_noise(mu, t)
            
            pred = diff_model(noisy, t)
            loss = torch.nn.functional.mse_loss(pred, noise)
            loss.backward()
            optimizer.step()
            
        diff_path = os.path.join(self.ckpt_dir, "diffusion.pt")
        torch.save(diff_model.state_dict(), diff_path)
        
        # 3. Generate Sample
        subprocess.run([
            "uv", "run", "python", "ml/sample_diffusion.py",
            "--vae-path", vae_path,
            "--diffusion-path", diff_path,
            "--output-dir", self.gen_dir,
            "--num-samples", "1",
            "--n-field", str(n_field),
            "--latent-dim", str(latent_dim),
            "--hidden-dim", "64",
            "--num-layers", "2",
            "--timesteps", "10"
        ], check=True)
        
        # Verify output exists
        self.assertTrue(os.path.exists(os.path.join(self.gen_dir, "gen_0000.sun")))

if __name__ == "__main__":
    unittest.main()
