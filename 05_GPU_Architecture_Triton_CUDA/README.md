# 🚀 Phase 05: GPU Silicon Mastery — Architecture, OpenAI Triton, Raw CUDA & FlashAttention

> **Prerequisites:** [Phases 00–04](../README.md) completed (Matrix/Attention math, PyTorch C++ extensions, CPU caches, Roofline model, and strided C++ pointers).  
> **Estimated Pace:** 5–7 Months.  
> **Big Idea (1 Sentence):** *A CPU is a Ferrari built to execute 1 complex thread with low latency; a GPU is a fleet of 10,000 buses built for massive throughput—and unlocking its true TFLOP/s requires orchestrating 32-thread SIMT warps, coalesced HBM loads, bank-conflict-free Shared Memory (`SRAM`) tiling, and online fused algorithms like `FlashAttention`.*

---

## 🎮 The Game Engine & Systems Connection
If you have ever looked at an **HLSL / GLSL / Vulkan Compute Shader**, welcome home—**CUDA IS compute shaders with full C++ pointers!**
* `SV_GroupID` in HLSL $\equiv$ `blockIdx` in CUDA.
* `SV_GroupThreadID` in HLSL $\equiv$ `threadIdx` in CUDA.
* `groupshared float tile[16][16]` in HLSL $\equiv$ `__shared__ float tile[16][16]` in CUDA!
* `GroupMemoryBarrierWithGroupSync()` in HLSL $\equiv$ `__syncthreads()` in CUDA!
* Even better: why do game shader programmers avoid branching `if (pixel_type == A) ... else ...`? **Warp Divergence!** Because 32 threads in a warp share a single instruction counter! Everything you learn here makes you unstoppable at **both** GPU ML kernels and GPU-driven game/physics engines.

*Environment Note — Your Google Colab High-Performance Workflow:* Do not waste days fighting Windows CUDA/Triton compiler mismatches locally. Use **Google Colab (`T4` / `L4` / `A100` Linux GPU runtimes)** as your dedicated GPU silicon lab while editing/archiving clean `.py` and `.cu` source files in **VS Code** & **Visual Studio**:
1. **Triton & PyTorch `load_inline`:** Run `@triton.jit` kernels directly in Colab cells, or compile raw `.cu` files via `%%writefile kernel.cu` + `!nvcc -O3 -arch=sm_75 ...` (or `!git pull` from your GitHub repo).
2. **Colab $\to$ Windows Nsight Profiling Trick:** Run `!ncu --set full -o my_kernel_report ./kernel` or `!nsys profile -o timeline python bench.py` inside Colab, download the tiny `.ncu-rep` / `.nsys-rep` file, and open it locally on Windows in the free **NVIDIA Nsight Compute / Nsight Systems GUI** to inspect hardware Roofline charts and SRAM bank conflicts!
3. **MATLAB Verification:** Export edge-case input tensors (`scipy.io.savemat`) to verify numerical accuracy of online softmax and tiled reductions against `float64` **MATLAB** ground truth.

---

## 📚 Primary Learning Resources for Phase 05

* **The Canonical Textbook:**
  * David B. Kirk, Wen-mei W. Hwu, Izzat El Hajj — *Programming Massively Parallel Processors: A Hands-on Approach (`PMPP`, 4th Edition, 2022)* *(Read Chapters 1–11 & 16 chapter-by-chapter!)*
* **Interactive GPU Puzzles (Do These Before Large Kernels — Super Fun!):**
  * Sasha Rush — [Triton Puzzles](https://github.com/srush/Triton-Puzzles) & [GPU Puzzles](https://github.com/srush/GPU-Puzzles) *(Visual diagram-based puzzles teaching thread/block indexing, shared memory, and pooling from zero!)*
* **World-Class Systems Walkthroughs & Lectures:**
  * [GPU MODE Lecture Series (`gpu-mode/lectures`)](https://github.com/gpu-mode/lectures) *(Practitioner lectures on PMPP, Triton, Profiling, Ring Attention, and Tensor Cores)*
  * Simon Boehm — [*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*](https://siboehm.com/articles/22/CUDA-MMM) *(Arguably the greatest single technical blog post in ML Systems history!)*
  * [OpenAI Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) *(Vector Add, Fused Softmax, Matmul, LayerNorm, Low-Memory Dropout, FlashAttention)*
  * [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
* **Seminal Papers (Read in Exact Order):**
  1. Philippe Tillet, H. T. Kung, David Cox (2019) — [*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
  2. Maxim Milakov & Natalia Gimelshein (NVIDIA, 2018) — [*Online normalizer calculation for softmax*](https://arxiv.org/abs/1805.02867) *(5-page gem that made FlashAttention possible!)*
  3. Tri Dao et al. (2022 / 2023) — [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135) & [*FlashAttention-2*](https://arxiv.org/abs/2307.08691)

---

## 🧱 Chronological Modules

### Module 5.1: GPU Silicon Architecture — SMs, Warps (SIMT) & The Memory Pyramid
*Understanding the physical layout of an NVIDIA die before writing a line of kernel code.*

* **Concepts Built From Scratch:**
  1. **Streaming Multiprocessors (`SMs`):** A GPU is an array of independent SMs (e.g., 40 SMs on T4, 108 SMs on A100, 132 SMs on H100). Thread **Blocks** are scheduled onto SMs; threads in *different* blocks cannot synchronize easily because blocks must be runnable in any order on 4 SMs or 132 SMs!
  2. **Warps (32 Threads in SIMT Lockstep) & Latency Hiding:**
     * Unlike a CPU that uses huge branch predictors and out-of-order caches to hide memory latency on 1 thread, an SM keeps **dozens of warps resident in massive Register Files simultaneously**—the instant Warp 0 stalls waiting 400 cycles for HBM memory, the warp scheduler switches to Warp 1 in **zero clock cycles**!
     * **Warp Divergence:** If threads `0..15` take `if` and threads `16..31` take `else`, the hardware executes *both* branches serially with masks, cutting throughput in half!
  3. **The GPU Memory Bandwidth & Capacity Pyramid (e.g., A100 80GB):**
     * **Registers** (~256 KB per SM): ~19+ TB/s, private per thread (*spill too many registers $\to$ drops Occupancy!*).
     * **Shared Memory / L1 Cache (`SRAM`)** (~164–228 KB per SM, ~20 MB total across chip): **~19 TB/s** on-chip programmer-managed scratchpad!
     * **L2 Cache** (40–50 MB): ~4–6 TB/s.
     * **Global Memory (`HBM` / VRAM)** (16–80 GB): **~1.5–3.3 TB/s** (~10x slower than SRAM!).
     * **PCIe Host $\leftrightarrow$ Device Bus:** **~16–64 GB/s** (~50x–100x slower than HBM! Never ping-pong tensors between CPU and GPU!).

---

### Module 5.2: Pythonic GPU Block Programming with OpenAI Triton
*Mastering tiled SRAM algorithms and operator fusion first without drowning in C++ boilerplate.*

* **Concepts Built From Scratch (`triton_kernels/`):**
  1. **SPMD Block Paradigm (`@triton.jit`):** In CUDA, you program a single *scalar thread*; in Triton, you program an entire **Block of elements** (`BLOCK_SIZE = 1024`), while Triton's compiler handles intra-block threads, vectorization, and shared-memory staging automatically!
  2. **Pointer Math & Boundary Masking:** `pid = tl.program_id(axis=0)`, `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`, `mask = offsets < n_elements`, `tl.load(ptr + offsets, mask=mask, other=-float('inf'))`.
  3. **Kernel Fusion (Beating Native PyTorch Eager by 4x!):**
     * Why is `torch.softmax(x, dim=-1)` faster than `exp(x - x.max()) / sum(...)`? Because PyTorch eager reads & writes HBM **5 separate times**!
     * Write **Fused Row-wise Softmax**, **Fused `RMSNorm` (Forward + Backward!)**, and **Fused `SwiGLU`** in Triton where an entire row loads into SRAM **once**, reduces in registers, and writes back **once**!
  4. **Autotuning (`@triton.autotune`):** Benchmarking across `BLOCK_SIZE` and `num_warps` configurations automatically and plotting achieved **GB/s vs Peak HBM Bandwidth**!
* **⚡ Practical Magic Lab (`triton_kernels/01_fused_softmax_and_rmsnorm.py`):**
  * Complete Sasha Rush's *Triton Puzzles*, then implement Fused Softmax & Fused RMSNorm (forward + backward wrapped in `torch.autograd.Function`), verify against PyTorch `float64`, and plot GB/s throughput showing your Triton kernel crushing eager PyTorch and matching/beating `torch.compile`!

---

### Module 5.3: Raw CUDA C++ Foundations, Streams & Global Memory Coalescing
*Dropping beneath Triton to command individual threads, warps, and PTX/SASS execution directly.*

* **Concepts Built From Scratch (`cuda_kernels/`):**
  1. **Host vs Device Lifecycle:** `__global__` (callable from CPU, runs on GPU), `__device__` (GPU helper), `__host__`, `cudaMalloc`, `cudaMemcpyAsync`, error-checking macro `CUDA_CHECK(err)`.
  2. **1D / 2D / 3D Coordinate Geometry:**
     $$\text{col} = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}, \qquad \text{row} = \text{blockIdx.y} \cdot \text{blockDim.y} + \text{threadIdx.y}$$
     Ceiling grid launch rule: `dim3 blocks((N + 255) / 256);`.
  3. **🔍 The #1 Rule of CUDA — Global Memory Coalescing:**
     * Remember `threadIdx.x` changes fastest within a 32-thread warp!
     * If Thread `t` reads `matrix[row * W + col]` where `col` maps to `threadIdx.x`, Threads `0..31` read **32 adjacent floats (128 contiguous bytes)**—merged by the GPU memory controller into **1 single 128-byte HBM burst transaction**!
     * If you accidentally map `row` to `threadIdx.x` (`matrix[threadIdx.x * W + blockIdx.x]`), Threads `0..31` read addresses `W * 4` bytes apart, issuing **32 separate uncoalesced memory transactions** and wasting **96% of your HBM bandwidth**!
  4. **Binding `.cu` Kernels to PyTorch (`load_inline` / `CUDAExtension`):** Extracting `tensor.data_ptr<float>()` on a `device="cuda"` PyTorch tensor—zero `cudaMemcpy` needed because PyTorch already allocated the HBM buffer!
* **⚡ Practical Magic Lab (`cuda_kernels/02_coalesced_vs_strided_transpose.cu`):**
  * Complete Sasha Rush's *GPU Puzzles*, then write a Matrix Transpose kernel 3 ways: (a) Naive uncoalesced write, (b) Shared-memory coalesced read + coalesced write (with bank conflicts), and (c) Padded `__shared__ float tile[32][33]`—measuring exact GB/s at each step!

---

### Module 5.4: Shared Memory `SRAM`, Bank Conflicts, Warp Shuffles & Parallel Prefix Scan
*How 32 threads cooperate at 19 TB/s without stepping on each other.*

* **Concepts Built From Scratch:**
  1. **Shared Memory & Synchronization (`__shared__`, `__syncthreads()`):** Race conditions (`read-after-write` and `write-after-read` hazards when loading next tile before all threads finish computing current tile!).
  2. **32 Shared Memory Banks:** Shared memory is striped across 32 hardware banks (4 bytes wide each). If all 32 threads in a warp access column `j` of `__shared__ float s[32][32]`, every element `s[0][j], s[1][j], ..., s[31][j]` lands in the **exact same bank**, serializing 1 read into **32 sequential cycles (32-way bank conflict)**! Declaring `s[32][33]` shifts each row by 1 bank—**0 bank conflicts**!
  3. **Warp-Level Primitives (`__shfl_down_sync`):** Threads inside the same 32-thread warp can read *directly from each other's registers* in 5 instructions without touching Shared Memory or `__syncthreads()`!
     ```cpp
     for (int offset = 16; offset > 0; offset >>= 1)
         val += __shfl_down_sync(0xffffffff, val, offset);
     ```
  4. **Parallel Reduction & Parallel Prefix Scan (Hillis-Steele / Blelloch):** Turning $O(N)$ sequential sum and cumulative prefix scan into $O(\log N)$ parallel trees! *(Remember Phase 01 Module 1.5? Parallel associative scan is the exact hardware primitive behind **Mamba / State-Space Models**, stream compaction, and GPU particle physics collision sorting!)*
* **⚡ Practical Magic Lab (`cuda_kernels/03_warp_reduction_and_scan.cu`):**
  * Implement a two-stage Warp-Shuffle Sum/Max Reduction + Blelloch Exclusive Prefix Scan in raw CUDA and verify against `torch.sum` and `torch.cumsum` on $N = 10^8$ floats!

---

### Module 5.5: Climbing the CUDA GEMM Ladder (From Naive to >85% of `cuBLAS`)
*The ultimate rite of passage for every GPU systems engineer.*

* **Work Step-by-Step Through Simon Boehm's Optimization Ladder (`cuda_kernels/04_sgemm_ladder.cu`):**
  1. **Kernel 1 (Naive Uncoalesced):** Mapping `x` to rows, `y` to cols (~0.3 TFLOPS — terrible!).
  2. **Kernel 2 (Global Memory Coalesced):** Mapping `threadIdx.x` to columns (~2 TFLOPS).
  3. **Kernel 3 (Shared Memory Block Tiling `32x32`):** Loading `As[32][32]` and `Bs[32][32]` into SRAM, cutting HBM traffic by $32\times$ (~4–6 TFLOPS—now bottlenecked by shared-memory `LDS` load instructions per FMA!).
  4. **Kernel 4 (1D & 2D Register Thread Tiling `TM=8, TN=8`):** Each thread computes an $8 \times 8 = 64$-element sub-patch of $C$ inside **registers** via outer products of `regA[8]` and `regB[8]`—slashing SRAM reads by $8\times$!
  5. **Kernel 5 (Vectorized `float4` 128-bit Loads + Double Buffering):** Overlapping next tile load with current tile compute!
  6. **Kernel 6 (Bonus — Tensor Cores `nvcuda::wmma`):** Using specialized $16 \times 16 \times 16$ matrix-multiply hardware units (`FP16`/`BF16` $\to$ `FP32` accumulate) to witness **5x–10x higher TFLOP/s than standard FP32 CUDA cores**!

---

### Module 5.6: Online Softmax & Implementing `FlashAttention` from Scratch
*Seeing Statistics, Numerical Analysis, and GPU Memory Hierarchy unite into modern AI's greatest systems breakthrough.*

* **Concepts Derived on Paper Before Coding:**
  1. **Why Standard Attention Hits a Memory Wall at Long Context ($N = 16{,}384$):**
     * $Q, K, V \in \mathbb{R}^{N \times d}$ ($d = 128$) are tiny, **but** attention matrix $S = Q K^\top \in \mathbb{R}^{N \times N}$ has $16{,}384^2 = 2.68 \times 10^8$ entries per head! Writing $S$ to HBM, reading $S$ to compute `Softmax` $P$, writing $P$ to HBM, reading $P$ to multiply $PV$ makes standard attention **catastrophically HBM bandwidth & capacity bound ($O(N^2)$ memory)**!
  2. **The Mathematical Obstacle:** How can you tile across columns of $K, V$ in SRAM when `Softmax` needs the global row maximum $m = \max_j S_{i,j}$ and global denominator $\ell = \sum_j e^{S_{i,j} - m}$ across *all* columns before multiplying by $V$?!
  3. **The Statistical/Algebraic Miracle — Online Safe Softmax Recurrence (Milakov & Gimelshein + Tri Dao):**
     Suppose after seeing tile 1 you have running max $m^{(1)}$, running sum $\ell^{(1)}$, and running output $O^{(1)}$. When tile 2 arrives with local max $m^{(2)}$, set $m_{\text{new}} = \max(m^{(1)}, m^{(2)})$ and **rescale** both $\ell^{(1)}$ and $O^{(1)}$ on the fly by $e^{m^{(1)} - m_{\text{new}}}$:
     $$\ell_{\text{new}} = e^{m^{(1)} - m_{\text{new}}}\ell^{(1)} + \sum_j e^{S^{(2)}_j - m_{\text{new}}}, \qquad O_{\text{new}} = \frac{\ell^{(1)} e^{m^{(1)} - m_{\text{new}}} O^{(1)} + e^{S^{(2)} - m_{\text{new}}} V^{(2)}}{\ell_{\text{new}}}$$
     **Zero approximation!** Exact bit-for-bit mathematical equivalence to full $N \times N$ Softmax in a **single streaming SRAM pass** with **$O(N)$ memory**!

---

### Module 5.7: Hardware Profiling (`nsys` & `ncu`) and `CUDA Graphs`
*Proving your kernel utilization directly from silicon hardware performance counters.*

* **Concepts Built From Scratch:**
  1. **Nsight Systems (`nsys profile`):** Timeline profiling—spotting CPU Python dispatch bubbles where the GPU sits completely idle between tiny kernels!
  2. **Nsight Compute (`ncu`):** Kernel-level Roofline chart, SM Compute Throughput %, Memory Throughput %, Occupancy %, Shared Memory Bank Conflicts count, and Warp Stall reasons (`Stall Long Scoreboard` = waiting on HBM!).
  3. **`CUDA Graphs` (`torch.cuda.CUDAGraph`):** Recording 500 small decode kernels into a single replayable hardware graph launched in **1 CPU call**, completely erasing Python CPU overhead!

---

## 📝 Required `notes/` Artifacts for Phase 05

1. `01_gpu_silicon_and_memory_map.md` — Diagram of GPU SMs, 32-thread Warps, Register File, 32-bank SRAM, L2, and HBM with exact latencies & bandwidths.
2. `02_sgemm_worklog.md` — Your own Simon-Boehm-style engineering worklog explaining what bottlenecked each GEMM kernel and how the next kernel broke that ceiling.
3. `03_online_softmax_and_flashattention_proof.md` — Full algebraic proof that the streaming online rescaling recurrence produces the exact same output $O$ as standard two-pass Softmax Attention, plus IO-complexity derivation ($O(N^2 d^2 / M)$ HBM accesses vs $O(N^2 + Nd)$).

---

## 🏆 Phase 05 Two-Part Boss Fight: `cuBLAS` Challenger + `Mini-FlashAttention`

**Location:** `05_GPU_Architecture_Triton_CUDA/`

1. **Part A (`cuda_kernels/boss_sgemm/`):** Benchmark all 5+ of your raw CUDA SGEMM kernels alongside your `@triton.jit` GEMM kernel and PyTorch `torch.matmul` (`cuBLAS`) across $N \in \{512, 1024, 2048, 4096\}$, outputting a publication-clean Matplotlib TFLOP/s curve + `torch.allclose` verification.
2. **Part B (`boss_flash_attention/`):** Implement **Tiled `FlashAttention` Forward Pass** in *both* OpenAI Triton and raw Shared-Memory CUDA C++ (bound via `Pybind11` / `load_inline`).
   * Verify numerical accuracy against naive `Softmax(Q @ K.T / sqrt(d)) @ V` (`atol=1e-3` in `FP16`/`FP32`).
   * Plot **Peak VRAM (MB) and Latency (ms)** across sequence lengths $N \in \{512, 1024, 2048, 4096, 8192, 16384\}$—watching naive PyTorch crash with `CUDA OutOfMemoryError` while your tiled `FlashAttention` cruises linearly in $O(N)$ memory!
