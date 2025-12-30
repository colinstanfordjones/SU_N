import torch
import numpy as np
import os

class SimpleVectorDB:
    """
    A simple brute-force vector database for indexing physics latent vectors.
    """
    def __init__(self, latent_dim=1024):
        self.vectors = []
        self.metadata = [] # stores (file, block_idx)
        self.latent_dim = latent_dim
        self.vectors_np = None

    def add(self, vector, meta):
        """Add a vector and its metadata to the database."""
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        self.vectors.append(vector.flatten())
        self.metadata.append(meta)

    def build_index(self):
        """Prepare the numpy array for searching."""
        if len(self.vectors) == 0: return
        self.vectors_np = np.array(self.vectors)
        
    def search(self, query_vector, k=5):
        """
        Search for the k nearest neighbors to the query vector.
        Returns: (metadata_list, distances)
        """
        if self.vectors_np is None:
            self.build_index()
            
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.detach().cpu().numpy()
        
        query_vector = query_vector.flatten()
        
        # Euclidean distance
        diff = self.vectors_np - query_vector
        dist = np.linalg.norm(diff, axis=1)
        
        # Get top k
        indices = np.argsort(dist)[:k]
        
        results = [self.metadata[i] for i in indices]
        distances = [dist[i] for i in indices]
        
        return results, distances

    def save(self, path):
        """Save the database to a .npz file."""
        np.savez(path, vectors=self.vectors_np, metadata=np.array(self.metadata, dtype=object))

    def load(self, path):
        """Load the database from a .npz file."""
        data = np.load(path, allow_pickle=True)
        self.vectors_np = data['vectors']
        self.metadata = data['metadata'].tolist()
        self.vectors = self.vectors_np.tolist()

if __name__ == "__main__":
    # Test DB
    db = SimpleVectorDB(latent_dim=4)
    db.add(np.array([1, 0, 0, 0]), {"id": "a"})
    db.add(np.array([0, 1, 0, 0]), {"id": "b"})
    db.add(np.array([1, 1, 0, 0]), {"id": "c"})
    db.build_index()
    
    res, dist = db.search(np.array([0.9, 0.1, 0, 0]), k=2)
    print(f"Results: {res}")
    print(f"Distances: {dist}")
