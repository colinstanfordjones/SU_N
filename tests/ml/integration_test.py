import unittest
import torch
import os
import shutil
import subprocess
import numpy as np
from ml.model import VAE
from ml.dataset import SunDataset
from ml.vector_db import SimpleVectorDB

class TestMLIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "tests/ml_temp"
        cls.data_dir = os.path.join(cls.test_dir, "data")
        cls.ckpt_dir = os.path.join(cls.test_dir, "checkpoints")
        cls.db_path = os.path.join(cls.test_dir, "test_db.npz")
        
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.ckpt_dir, exist_ok=True)
        
        # 1. Generate Synthetic Data using the Zig tool
        # We assume 'zig build generate-dataset' has been built or we call the binary directly
        # For CI/Test environment, we'll try to run via zig build
        print("Generating synthetic dataset...")
        subprocess.run(
            ["zig", "build", "generate-dataset", "--", "5", cls.data_dir],
            check=True,
            capture_output=True
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_dataset_loading(self):
        dataset = SunDataset(self.data_dir)
        self.assertGreater(len(dataset), 0)
        
        sample = dataset[0]
        # Shape should be (N_field*2, 16, 16, 16, 16)
        # N=1 -> N_field=1 -> 2 channels
        self.assertEqual(len(sample.shape), 5)
        self.assertEqual(sample.shape[0], 2) # Real, Imag
        self.assertEqual(sample.shape[1], 16)

    def test_vae_convergence(self):
        """Train for a few epochs and check if loss decreases"""
        device = torch.device("cpu") # Force CPU for testing stability
        
        dataset = SunDataset(self.data_dir)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        
        model = VAE(n_field=1, latent_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        initial_loss = float('inf')
        final_loss = 0
        
        model.train()
        
        # Train for 5 epochs
        for epoch in range(5):
            epoch_loss = 0
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(batch)
                
                # Simple MSE + KLD
                mse = torch.nn.functional.mse_loss(recon, batch, reduction='sum')
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = mse + kld
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataset)
            if epoch == 0:
                initial_loss = avg_loss
            final_loss = avg_loss
            print(f"Epoch {epoch}: Loss {avg_loss}")

        self.assertLess(final_loss, initial_loss, "Loss should decrease after training")

    def test_vector_db_workflow(self):
        # Create a dummy DB
        db = SimpleVectorDB(latent_dim=4)
        vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        
        db.add(vec1, {"id": 1})
        db.add(vec2, {"id": 2})
        db.build_index()
        db.save(self.db_path)
        
        # Reload
        db2 = SimpleVectorDB(latent_dim=4)
        db2.load(self.db_path)
        
        # Search for something close to vec1
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results, dists = db2.search(query, k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)
        self.assertLess(dists[0], 0.2)

if __name__ == "__main__":
    unittest.main()
