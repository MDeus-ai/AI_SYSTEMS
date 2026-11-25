# 🏛️ Phase 07: Frontier Research Capstones (Unmatched Mastery & Publication-Grade Portfolio)

> **Prerequisites:** [Phases 00–06](../README.md) completed.  
> **Estimated Pace:** 4–6+ Months per Capstone (Ongoing Research Career Launchpad).  
> **Big Idea (1 Sentence):** *Transition from Student to Frontier ML Systems Researcher by identifying a real mathematical or algorithmic bottleneck where off-the-shelf frameworks choke, co-designing the statistical math and the custom C++/Triton/CUDA silicon execution together, and publishing an indisputable, paper-grade open-source engine.*

---

## 🎯 What Separates a Toy GitHub Repo from a Frontier Lab Offer?

When a Research Lead at DeepMind, OpenAI, Anthropic, NVIDIA, Meta FAIR, or a Professor at Mila / Vector / CMU / Waterloo opens a candidate's GitHub, 99% of portfolios show high-level wrapper scripts (`AutoModelForCausalLM.from_pretrained(...)`).

Your Capstone repository will look like a **peer-reviewed systems & machine learning research paper accompanied by reproducible, hardware-verified C++/CUDA/Triton code**:
1. **A Real Bottleneck:** Standard PyTorch/JAX is already near-optimal at dense `cuBLAS` `matmul`. You target regimes where standard frameworks fail: sequential MCMC/ODE dynamics, irregular/structured covariance matrices, memory-bound long-context recurrences, custom second-order preconditioners, or CPU-bottlenecked RL physics simulators.
2. **Mathematical Ground Truth:** Exact analytical proofs + double-precision (`float64` / SymPy / MATLAB) numerical equivalence (`assert_close`).
3. **Hardware-Aware Co-Design:** Explicit Roofline analysis, SRAM tiling / warp reductions / memory coalescing, Nsight Compute hardware counter receipts, and clean ablation plots.

---

## 🧭 Four Frontier Capstone Tracks (Choose Your Flagship—or Combine Two!)

### Track A: Statistical Silicon — Hardware-Accelerated Bayesian Inference or Second-Order Optimization
*Leveraging your 3.99 BSc in Statistics at the bleeding edge of GPU computing.*
* **Project Ideas:**
  1. **GPU-Fused No-U-Turn / Hamiltonian Monte Carlo (`CUDA-HMC`):** Sequential leapfrog steps and dynamic trajectory trees suffer massive Python/kernel-launch overhead in PyTorch/Stan. Build a warp-parallelized, shared-memory-cached C++/CUDA HMC engine with custom analytical log-posterior gradients, benchmarking Effective Sample Size per second (`ESS/s`) across 10,000 parallel chains against Stan/Pyro/BlackJAX.
  2. **Tiled Shared-Memory Structured Gaussian Process (`GP`) Engine:** Exact GPs scale as $O(N^3)$ compute and $O(N^2)$ memory. Combine **Blackbox Matrix-Matrix Conjugate Gradients (`BBMM` / GPyTorch math)** with a **Fused On-the-Fly Tiled Kernel-Matrix-Vector CUDA Kernel** (computing $K(x_i, x_j)v_j$ inside SRAM tiles without *ever* materializing the $N \times N$ covariance matrix in HBM—just like FlashAttention for Gaussian Processes!).
  3. **Fused Kronecker / Newton-Schulz Matrix Optimizer (`CUDA-Muon` / `K-FAC`):** Implement a fused Triton/CUDA Newton-Schulz orthogonalization or Kronecker-factored curvature preconditioner and run controlled wall-clock-to-target-validation-loss scaling experiments against `AdamW`.

### Track B: Architecture-Hardware Co-Design — Novel Sequence / Attention / State-Space Kernels
*Co-designing sequence modeling equations around SRAM and Tensor Core constraints.*
* **Project Ideas:**
  1. **Custom Hardware-Aware Linear / Gated DeltaNet / Selective State-Space (`Mamba`) Kernel:** Derive the associative parallel scan backward pass on paper, implement fused forward + backward + activation-recomputation kernels in Triton and raw CUDA (using warp shuffles & SRAM tiling), verify gradients to $10^{-5}$, and benchmark throughput & perplexity up to $N = 65{,}536$ context length!
  2. **Sparse / Block-Sparse / MLA (Multi-Head Latent Attention) Kernel Suite:** Implement DeepSeek-style low-rank KV latent compression (`MLA`) fused with `RoPE` and Paged KV-Cache, quantifying exact HBM bandwidth savings during long-context decoding.

### Track C: Simulation & Embodied AI — Differentiable or Massively Parallel GPU Game/Physics Engine
*Uniting your Game Engine passion and RL Systems engineering into a flagship engine.*
* **Project Ideas:**
  1. **Batched C++20/CUDA Multi-Agent Rigid-Body & Raycast Game Engine:** Build a complete deterministic GPU-native 2D/3D game physics engine (broadphase spatial hash grid in parallel prefix scan + narrowphase impulse solver + batched BVH/grid raycasting sensors in CUDA) capable of stepping **32,768 multi-agent competitive worlds at >5,000,000 steps/sec**, training emergent cooperative/competitive strategies via self-play `PPO` on a single GPU in minutes!
  2. **Differentiable Physics Engine (`DiffPhys`):** Build a custom reverse-mode differentiable rigid-body / spring-mass soft-body simulator in C++/CUDA where analytical physics gradients $\frac{\partial s_{t+1}}{\partial a_t}$ propagate backward through time directly into policy weights—comparing sample efficiency against model-free `PPO` (`REINFORCE` variance vs analytical pathwise physics gradients!).

### Track D: Extreme Inference & Compression — Standalone C++20/CUDA Speculative Quantized Engine
*Building a self-contained production/edge engine from bare metal.*
* **Project Ideas:**
  1. Zero-dependency C++20/CUDA inference engine supporting `mmap` GGUF/safetensors loading, custom fused `W4A16` / `FP8` kernels, `CUDA Graphs`, `PagedAttention` continuous batching, and tree-based **Speculative Decoding (`Medusa` / `EAGLE`-style draft heads)** with rigorous Roofline efficiency benchmarks.

---

## 🔬 The 6-Stage Research Execution Protocol

Execute your capstone across your unified 4-pillar workstation (**MATLAB** for symbolic/statistical ground truth $\to$ **VS Code** for Python/PyTorch baselines & `pytest` harnesses $\to$ **Visual Studio** for C++20 engine architecture & memory debugging $\to$ **Google Colab** + **Nsight** for Triton/CUDA silicon execution and Roofline benchmarking) inside `07_Frontier_Research_Capstones/`:

1. **Stage 1 — Proposal & Mathematical Specification (`project_proposals/01_spec.md`):**
   * State the exact mathematical problem in clean LaTeX.
   * Calculate back-of-the-envelope **FLOPs, Bytes Moved, and Arithmetic Intensity** *before* writing code (Google DeepMind *Scaling Book* style!) to predict theoretically whether your idea is compute-bound or bandwidth-bound.
2. **Stage 2 — Untouchable Ground Truth Reference:**
   * Implement the cleanest, simplest `float64` reference in pure Python/NumPy/PyTorch—and cross-verify core equations in [99_Verification_and_Papers/matlab_verification/](../99_Verification_and_Papers/matlab_verification/) (MATLAB / `SymPy` / `mpmath`) to 8+ decimal places. Save deterministic test fixtures to disk.
3. **Stage 3 — Baseline Profiling & Amdahl Proof:**
   * Run `torch.profiler` and `nsys` on the standard PyTorch/JAX baseline. Capture Flame Graphs and Nsight Roofline screenshots proving *exactly* why standard frameworks stall (e.g., *"87% of runtime is spent reading/writing intermediate $O(N^2)$ tensors to HBM at 82% memory bandwidth saturation while SM compute sits at 6%"*).
4. **Stage 4 — Progressive Kernel Engineering (Triton $\to$ Raw C++/CUDA):**
   * Prototype block logic and masking rapidly in OpenAI Triton first.
   * Engineer the raw C++20 / CUDA `.cu` implementation step-by-step (Coalescing $\to$ Shared Memory Tiling $\to$ Bank-Conflict Padding $\to$ Warp Shuffles / Register Tiling / Vectorized `float4` / Tensor Cores).
   * Wrap seamlessly into PyTorch/JAX via `Pybind11` / `torch.library` custom ops with analytical backward kernels.
5. **Stage 5 — Exhaustive Verification & Ablation Suite:**
   * Automated `pytest` suite asserting forward & backward numerical equivalence across edge-case shapes, non-power-of-two lengths, and precisions (`FP32`, `BF16`, `FP16`).
   * Step-by-step ablation benchmarks measuring latency (ms), peak VRAM (MB), achieved GB/s, achieved TFLOP/s, and % of hardware Roofline peak as each optimization is turned on!
6. **Stage 6 — The Frontier Technical Report (`README.md` + ArXiv-Style Writeup):**
   * Structure your Capstone's main `README.md` like a crisp DeepMind / Meta FAIR systems paper:
     1. **Abstract & Headline Result** *(e.g., "28.4x speedup and 14x memory reduction over PyTorch 2 `torch.compile` on exact GP inference with bit-verified gradients")*
     2. **Mathematical Formulation & Statistical Motivation**
     3. **Roofline Bottleneck Analysis of Standard Frameworks**
     4. **Hardware-Aware Algorithm & SRAM Tiling Architecture** *(with clean diagrams)*
     5. **Experimental Verification, Scaling Curves & Nsight Counter Analysis**
     6. **Limitations, Failure Modes & Future Research Directions** *(honest scientific rigor builds instant trust with senior researchers!)*

---

*When you finish Phase 07, you will not merely "know how to use AI tools." You will understand every bit from IEEE-754 mantissas, eigenvalue geometry, KL divergence, and Transformer scaling variances all the way down through C++20 Arena allocators, compiler IRs, warp shuffles, SRAM bank conflicts, and multi-GPU collectives. That is unmatched mastery.*
