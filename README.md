# su_n: Distributed Adaptive Mesh Refinement Engine

`su_n` is a high-performance, domain-agnostic Adaptive Mesh Refinement (AMR) engine written in Zig. It provides a robust infrastructure for distributed multi-scale simulations, featuring dynamic load balancing, a geometric multigrid solver, and strict flux conservation.

**Status:** Alpha (Core Architecture Complete)

## 🚀 Key Features

*   **Distributed Parallelism:** Hybrid MPI + Threading architecture for scalable execution across nodes.
*   **Linear Octree Storage:** Blocks stored in Morton (Z-order) curve order for optimal cache locality.
*   **Geometric Multigrid (GMG):** Full V-Cycle solver support for elliptic partial differential equations.
*   **Strict Conservation:** Flux registers ensure numerical conservation at coarse-fine refinement boundaries (Refluxing).
*   **Domain Agnostic:** Physics kernels are decoupled from mesh logic via a type-safe `Frontend` interface.
*   **Zero-Allocation Hot Paths:** Heavy use of memory arenas and pre-allocated buffers to minimize runtime overhead.

## 🛠️ Architecture

`su_n` separates infrastructure from physics using Zig's comptime generics.

*   **`AMRTree`**: The core linear octree. Manages block lifecycle, neighbor lookups, and load balancing.
*   **`DistExchange`**: A unified communication layer handling both local memory copies (shared memory) and MPI message passing (distributed memory).
*   **`FluxRegister`**: Tracks fluxes across resolution boundaries to enforce conservation laws.
*   **`GaugeTree`**: An extension for Lattice Gauge Theory (QCD), managing link variables and covariant derivatives.

## ⚡ Performance

Benchmarks on a single node show minimal overhead for infrastructure management.

*   **Evolution Overhead:** ~1ms per step (2-block test case).
*   **Memory:** Compact block storage with structure-of-arrays layout for fields.

## 📦 Building & Running

**Prerequisites:**
*   Zig 0.13+
*   MPI (optional, for distributed runs)

**Run the Benchmark:**
```bash
zig build bench-flux-benchmark
```

**Run Tests:**
```bash
zig build test
```

## 📄 License

MIT License