# Machine Learning Integration Specification

This document outlines the architecture for integrating Machine Learning (ML) with the `su_n` physics engine to enable Inverse Design and Semantic Search of physical states.

## Overview

The integration bridges the gap between the high-precision AMR physics engine and probabilistic AI models. It uses a **Variational Autoencoder (VAE)** to map complex 4D lattice states into a compact **Latent Vector Space**, enabling efficient storage, search, and generation.

## Components

### 1. Data Bridge (`src/platform/checkpoint/`)
- **Format:** `.sunc` checkpoint snapshots.
- **Data Layout:** Header + raw `AMRTree` blocks + `field_slots` + full `FieldArena` storage/free list + optional gauge links.
- **Purpose:** Exact snapshots for restart, validation, and ML dataset extraction (block-level slicing happens in tooling).

### 2. Dataset Generation (`tools/generate_dataset.zig`)
- **Workflow:**
  1. Initialize randomized parameters (e.g., Coulomb centers, charges).
  2. Seed wavefunction with noise.
  3. Evolve in imaginary time (`evolveImaginaryTimeAMR`) to induce physical correlations (ground state relaxation).
  4. Write `.sunc` checkpoints and extract valid blocks for datasets.
- **Volume:** Generates massive datasets of valid Hamiltonian eigenstates.

### 3. 4D VAE Architecture (`ml/model.py`)
A specialized Autoencoder for 4D spacetime blocks ($T, X, Y, Z$).

- **Input Tensor:** `(Batch, N_field * 2, 16, 16, 16, 16)`
  - Time dimension $T=16$ is treated as depth or folded into channels depending on 3D/4D conv strategy.
- **Encoder:**
  - 3D Convolutional layers reducing spatial dimensions ($16^3 	o 2^3$).
  - Flattens to Latent Vector $\mu, \sigma$ (default dim=1024).
- **Latent Space:** Gaussian distribution $N(0, 1)$ enforced by KL Divergence loss.
- **Decoder:**
  - Transposed 3D Convolutions expanding latent vector back to $16^4$ block.
- **Loss Function:** MSE (Reconstruction) + $\beta \cdot$ KLD (Regularization).

### 4. Latent Diffusion Model (`ml/diffusion.py`)
To avoid the "averaging" artifacts of VAEs, we use Latent Diffusion to generate high-fidelity states.
- **Architecture:** Residual MLP with sinusoidal time embeddings.
- **Process:**
  1. **Forward:** Add Gaussian noise to latent vectors $z$ over $T=1000$ steps.
  2. **Backward:** Denoising network predicts noise $\epsilon$ given noisy latent $z_t$ and step $t$.
- **Scripts:**
  - `ml/train_diffusion.py`: Trains the denoising network on encoded datasets.
  - `ml/sample_diffusion.py`: Generates new physics blocks from pure noise.

### 5. Vector Database (`ml/vector_db.py`)
- **Storage:** Simple in-memory or file-based (`.npz`) store for prototype; scalable to Faiss/Pgvector.
- **Content:**
  - **Vector:** 1024-float latent representation of a physics block.
- **Metadata:** `{file_path, offset, level, origin}` pointing to the source `.sunc` file.
- **Query:** Euclidean (L2) distance search to find "nearest neighbor" physical states.

## Workflows

### A. Semantic Search ("Physics Google")
1. **Index:** Run `ml/encode_dataset.py` on a massive library of simulations.
2. **Query:** "Find a state that looks like *this* specific vortex configuration."
3. **Retrieval:** Encoder $\to$ Vector DB Search $\to$ Return Metadata.
4. **Analysis:** Load raw `.sunc` file from metadata for precise physics analysis.

### B. Inverse Design (Generative Loop)
1. **Dream:** AI model (e.g., Diffusion) generates a Latent Vector $z$ targeting specific properties (e.g., specific energy density).
2. **Decode:** VAE Decoder maps $z \to \psi_{guess}$ (approximate field).
3. **Verify:**
   - Load $\psi_{guess}$ into `su_n`.
   - Run `Hamiltonian.measureEnergy()`.
   - **Refine:** Run short `evolveImaginaryTime()` to snap $\psi_{guess}$ to the nearest valid eigenstate.
4. **Learn:** Add the refined state back to the training set to improve the VAE.

## Directory Structure

```
ml/
├── dataset.py        # PyTorch Dataset for .sunc files
├── model.py          # 4D VAE Architecture
├── train.py          # Training loop
├── vector_db.py      # Vector storage and search
├── encode_dataset.py # Bulk indexing script
└── checkpoints/      # Saved models
```
