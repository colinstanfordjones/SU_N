# AI & Verification Workflow

This document outlines the workflows for using the `su_n` engine's AI capabilities for Physics, Material Science, and Quantum Error Correction (QEC).

## 1. The "Hallucinate-Check-Refine" Loop (Inverse Design)

**Goal:** Discover physical states that meet specific properties (e.g., Energy < E_target, specific topological charge) without exhaustive simulation.

**Workflow:**
1.  **Hallucinate (Generation):**
    *   Input: Random noise vector $z \sim N(0, 1)$.
    *   Process: `ml/sample_diffusion.py` generates a candidate latent vector $z_{gen}$ via Denoising Diffusion.
    *   Decode: VAE Decoder maps $z_{gen} \to \psi_{guess}$ (16^4 complex field).
    *   *Speed:* Milliseconds.

2.  **Check (Validation):**
    *   Input: $\psi_{guess}$.
    *   Process: `HamiltonianAMR.measureEnergy()`.
    *   Filter: Discard if $E > E_{threshold}$ or if conservation laws (charge/color) are grossly violated.

3.  **Refine (Relaxation):**
    *   Input: $\psi_{guess}$.
    *   Process: Run `HamiltonianAMR.evolveImaginaryTimeAMR()` for a short burst (e.g., 50 steps).
    *   Goal: Snap the "dreamt" state to the nearest valid eigenstate (removing non-physical artifacts).
    *   Output: $\psi_{final}$.

4.  **Index (Learning):**
    *   Input: $\psi_{final}$.
    *   Process: Encode $\psi_{final} \to z_{final}$ via VAE Encoder.
    *   Store: Add $z_{final}$ to the Vector DB to improve future searches.

## 2. Semantic Search (Anomaly Detection)

**Goal:** Find rare, high-value events in massive simulation datasets (e.g., QGP instantons, BEC vortices).

**Workflow:**
1.  **Indexing:**
    *   Run massive simulation (e.g., Thermal QCD).
    *   Every $N$ steps, checkpoint state $\psi_t$ to `.sunc`.
    *   Encode $\psi_t \to z_t$ and store in Vector DB with metadata (Temp, Coupling).

2.  **Anomaly Detection:**
    *   Query: `VectorDB.find_outliers(k=10)`.
    *   Logic: Find states with low probability density $P(z)$ under the VAE prior, or large distance from cluster centers.
    *   Output: List of timestamps $t$ containing potential anomalies.

3.  **Analysis:**
    *   Load specific blocks $\psi_t$.
    *   Visualize/Analyze with `su_n` physics tools.

## 3. Quantum Error Correction (Toric Code Verification)

**Goal:** Identify logical errors (topological defects) in a quantum memory simulation.

**Mapping:**
*   **Lattice:** Toric code ground state $\leftrightarrow$ `su_n` vacuum.
*   **Errors:** Bit-flips / Phase-flips $\leftrightarrow$ Gauge excitations.
*   **Logical Error:** Non-trivial loop $\leftrightarrow$ Topological Charge $Q \neq 0$.

**Workflow:**
1.  **Simulate:** Evolve system with noise (simulating qubit errors).
2.  **Measure:** Calculate Topological Charge $Q$ (Wilson Loop parity).
3.  **Detect:**
    *   If $Q$ changes, a logical error occurred.
    *   Use Vector DB to find *other* states with similar error syndromes ($z \approx z_{error}$). 
4.  **Verify:**
    *   Check if the Decoder (AI) can correct this syndrome.
    *   If AI fails but Physics Engine confirms stable error, it is a "Fooling Set" (Adversarial Example for the QEC code).

## Future Requirements

To fully enable these workflows, the engine requires:
1.  **Metadata:** `.sunc` files must store $T$, $g$, $m$, and $Q$.
2.  **Non-Linearity:** Gross-Pitaevskii term $g|\psi|^2\psi$ for BECs.
3.  **Thermal Driver:** Evolution with periodic boundary conditions in time ($\beta = 1/T$). 
