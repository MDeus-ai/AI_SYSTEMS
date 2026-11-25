# 🌐 Phase 06: Compilers, Quantization, LLM Serving, Distributed Scale & C++/CUDA RL Engines

> **Prerequisites:** [Phases 00–05](../README.md) completed (Deep Learning theory, C++20 DOD, custom framework internals, and Triton/CUDA kernel mastery).  
> **Estimated Pace:** 5–7 Months.  
> **Big Idea (1 Sentence):** *Single-kernel speed is only half the frontier—modern ML Systems Research happens where **ML Compilers** automate kernel fusion, **Quantization & Serving Engines** (`vLLM` / `llama.cpp`) break the memory-bandwidth wall of inference, **Distributed Parallelism** (`ZeRO`/`FSDP`/`Megatron`/`Ring Attention`) turns 1,000 GPUs into one supercomputer, and **GPU-Native RL Physics Engines** simulate 10,000 game worlds at >1,000,000 steps per second.*

> **🛠️ Phase 06 Tooling Setup:**
> * **Google Colab (`T4` / `A100`):** Primary execution lab for `torch.compile` (`TorchInductor` Triton codegen dumps), `W4A16` quantized kernels, `PagedAttention`, `TinyInfer` CUDA execution, and `HyperEnv` GPU RL stepping.
> * **Visual Studio 2022 (`MSVC`) + VS Code:** Architect and memory-debug the C++20 core of `TinyInfer` (`mmap` weight loader, tokenizer, arena KV-cache) and `HyperEnv` (`SoA` physics & collision solver) locally before deploying CUDA kernels to Colab.
> * **MATLAB:** Verify speculative decoding rejection-sampling distributions ($\chi^2$ / KL tests) and quantization error bounds.
> * **Kaggle (`2x T4` Multi-GPU — Module 6.4 Only):** Because a standard Colab runtime exposes 1 GPU, spin up a free Dual-T4 Kaggle notebook specifically for Module 6.4 when testing real multi-GPU `NCCL` `Ring-AllReduce`, `ZeRO`/`FSDP`, and `Megatron` Tensor Parallelism!

---

## 📚 Primary Learning Resources for Phase 06

* **ML Compilers:**
  * Tianqi Chen et al. — [*Machine Learning Compilation* (`mlc.ai`)](https://mlc.ai/) *(Free online book & video lectures)*
  * Edward Z. Yang / Jason Ansel et al. (2024) — *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation* (`TorchDynamo` + `AOTAutograd` + `TorchInductor`)
* **Scaling, Distributed Systems & LLM Serving (Two Absolute Masterpieces — Both Free!):**
  * Google DeepMind (Austin et al.) — [*How To Scale Your Model* (`jax-ml/scaling-book`)](https://jax-ml.github.io/scaling-book/) *(Chapters on Sharding, Collectives, LLaMA math, Training, and Inference—do the worked math exercises!)*
  * Nouamane Tazi et al. (Hugging Face) — [*The Ultra-Scale Playbook: Training LLMs on GPU Clusters*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) *(Interactive visualizations of Data, ZeRO, Tensor, Pipeline, and Context Parallelism!)*
  * Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) *(Systems, Parallelism & Scaling Law lectures)*
* **Seminal Systems Papers (Read in Phase 06):**
  1. Rajbhandari et al. (2020) — [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054)
  2. Shoeybi et al. (2019) — [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053)
  3. Kwon et al. (SOSP 2023) — [*Efficient Memory Management for Large Language Model Serving with PagedAttention (`vLLM`)*](https://arxiv.org/abs/2309.06180)
  4. Leviathan, Kalman, Matias (ICML 2023) — [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192) *(Pure probability acceptance/rejection brilliance!)*
* **Codebases to Read Like Novels:**
  * Andrej Karpathy — [`llm.c` (Training GPT-2 in pure C & CUDA without PyTorch!)](https://github.com/karpathy/llm.c)
  * Georgi Gerganov — [`llama.cpp` / `ggml`](https://github.com/ggml-org/llama.cpp)
  * Shacklett et al. (Stanford, SIGGRAPH 2023) — [*An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation (`Madrona`)*](https://madrona-engine.github.io/) + [CleanRL (`vwxyzjn/cleanrl`)](https://github.com/vwxyzjn/cleanrl)

---

## 🧱 Chronological Modules

### Module 6.1: ML Compilers & Automatic Graph Optimization (`torch.compile`, XLA & TVM)
*How a compiler looks at your naive Python math and automatically writes the fused Triton kernels you hand-crafted in Phase 05.*

* **Concepts Built From Scratch:**
  1. **The PyTorch 2 Compiler Stack:**
     * `TorchDynamo`: Intercepts Python C-eval frame bytecode dynamically to trace `torch.fx.Graph`s safely (handling *"graph breaks"* when hitting unsupported Python control flow).
     * `AOTAutograd`: Traces *both* the forward and backward graphs ahead-of-time so the compiler can optimize/fuse backward kernels too—and choose **Activation Checkpointing (Rematerialization)** via min-cut graph algorithms (trading 33% extra compute FLOPs to slash activation VRAM by $O(\sqrt{L})$!).
     * `TorchInductor`: Lowers `ATen` primitives to loop-level IR and emits actual Python **Triton GPU code** and **OpenMP/SIMD C++ CPU code**!
  2. **Reading Generated Compiler Code:** Running `TORCH_LOGS="output_code" python script.py` to inspect the *exact* Triton kernel `TorchInductor` wrote for your model—and spotting when the compiler fails to fuse an op so you know when to write a custom CUDA kernel!
* **⚡ Practical Magic Lab (`quantization_and_inference/01_inspect_inductor_and_checkpointing.py`):**
  * Trace a custom Transformer block with `torch.fx`, dump `TorchInductor`'s generated Triton code, and implement **Selective Activation Checkpointing** from scratch—measuring the exact VRAM savings vs runtime overhead on a 24-layer model!

---

### Module 6.2: Low-Precision Arithmetic & Quantization Kernels (`BF16`, `FP8`, `INT8`, `INT4`)
*How moving 4x fewer bits across the HBM memory bus makes a 70B LLM run 4x faster on a consumer GPU.*

* **Concepts Built From Scratch (`quantization_and_inference/`):**
  1. **Mixed-Precision Training Anatomy:** Master `FP32` master weights + `FP32` optimizer moments + `BF16`/`FP8` forward/backward activations/GEMMs (and why `FP16` needed dynamic `GradScaler` while `BF16` usually doesn't).
  2. **Post-Training Quantization (`PTQ`) Math:**
     * Symmetric AbsMax quantization: $s = \frac{\max |W|}{2^{b-1}-1}$, $W_q = \text{clamp}\left(\text{round}\left(\frac{W}{s}\right), -127, 127\right)$, $\hat{W} = s \cdot W_q$.
     * **🔍 Why Naive Per-Tensor Quantization Fails on LLMs (Emergent Outlier Channels!):** Above ~6B parameters, Transformers develop specific feature channels with magnitudes $100\times$ larger than all others (`LLM.int8()` Dettmers et al. discovery!). One outlier stretches $s$ for the whole matrix, squashing 99.9% of weights to `0`! Solution: **Per-Channel / Group-Wise (`group_size=128`) Quantization** + Activation-Aware Scaling (`AWQ`)!
  3. **Bit-Packing Two `INT4` Weights into One `uint8` Byte** (`packed = (w_high << 4) | (w_low & 0x0F)`) and **Fused Weight-Only Dequant-GEMV (`W4A16`)**:
     * *Trap:* If you dequantize `INT4` $\to$ `FP16` into a full HBM buffer and then call `matmul`, you read `INT4` + write `FP16` + read `FP16` = **slower than unquantized `FP16`!**
     * *Systems Fix:* Load packed `uint8` from HBM into GPU registers, unpack bitwise `(byte >> 4)` **inside registers**, multiply by `FP16` activation immediately! Now HBM traffic drops by a true **$4\times$**!
* **⚡ Practical Magic Lab (`quantization_and_inference/02_w4a16_fused_triton_gemv.py`):**
  * Pack an `FP16` weight matrix into group-wise `INT4` (`uint8` storage + `FP16` group scales), write a **Fused `W4A16` Dequant-MatVec Triton/CUDA kernel**, verify perplexity/MSE impact, and prove a **>3x latency speedup** at batch size 1 over `FP16` `cuBLAS`!

---

### Module 6.3: LLM Inference & Serving Engines (`Prefill` vs `Decode`, `PagedAttention` & Speculative Decoding)
*Deconstructing `vLLM`, `SGLang`, `TensorRT-LLM`, and `llama.cpp` from pure Roofline & Probability first principles.*

* **Concepts Built From Scratch:**
  1. **The Two Totally Different Worlds of LLM Inference:**
     * **Prefill Phase** (processing prompt of $T=2048$ tokens in parallel): Matrix-Matrix Multiply (`GEMM`), high Arithmetic Intensity $\implies$ **Compute-Bound**!
     * **Decode Phase** (generating 1 token at a time for $B=1$ user): Matrix-Vector Multiply (`GEMV`), reading **100% of model weights + entire KV-Cache** from HBM just to emit 1 token ($\text{AI} \approx 1$ FLOP/Byte!) $\implies$ **100% Memory-Bandwidth Bound**!
     * *Sanity-Check Formula Every Systems Researcher Knows Cold:* For `B=1` decode, theoretical max tokens/sec $\approx \frac{\text{HBM Bandwidth (GB/s)}}{\text{Model Size in Bytes (GB)}}$! (e.g., 14 GB `FP16` 7B model on 900 GB/s GPU cannot physically exceed $\frac{900}{14} \approx 64$ tok/s without quantization or speculative decoding!).
  2. **Exact KV-Cache Memory Math:**
     $$\text{KV Bytes per Token} = 2 \text{ (K \& V)} \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
  3. **`PagedAttention` (`vLLM`):** Why pre-allocating `max_seq_len = 32768` contiguous KV buffers wastes **60–80% of VRAM** to internal/external fragmentation! Apply OS Virtual Memory Paging from Phase 03: chop KV-cache into fixed 16-token **Physical Blocks** mapped via a per-request **Block Table** so a custom gather-attention kernel reads non-contiguous blocks with <4% waste, enabling massive **Continuous Batching**!
  4. **Speculative Decoding (Your Statistics Crown Jewel!):**
     * A tiny 100M "Draft" model proposes $\gamma = 5$ tokens fast; the huge 70B "Target" model scores all 5 tokens **in parallel in 1 single GEMM forward pass** (costing almost the same wall-clock time as 1 decode step because `B=1` was bandwidth-starved!).
     * **Prove on paper** why accepting draft token $x$ with probability $\min\left(1, \frac{p_{\text{target}}(x)}{q_{\text{draft}}(x)}\right)$ and resampling rejections from $\text{norm}(\max(0, p_{\text{target}} - q_{\text{draft}}))$ guarantees **mathematically 100% identical output distribution to sampling purely from $p_{\text{target}}$** while running **2x–3x faster**!
* **⚡ Practical Magic Lab (`quantization_and_inference/03_paged_kv_and_speculative_decoding.py`):**
  * Implement exact Lossless Speculative Decoding between a 2-layer draft Transformer and a 12-layer target Transformer; verify empirically via a $\chi^2$ / KL test over 50,000 samples that the token distribution matches pure target sampling while measuring wall-clock speedup!

---

### Module 6.4: Distributed Training & Multi-GPU Communication Math
*How frontier labs train 1-Trillion-parameter models across 16,384 GPUs without drowning in network latency.*

* **Concepts Built From Scratch (`distributed_labs/`):**
  1. **The `16Ψ` Memory Accounting Rule (Mixed-Precision `AdamW` with $\Psi$ Parameters):**
     * `BF16` Parameters: **$2\Psi$ bytes**
     * `BF16` Gradients: **$2\Psi$ bytes**
     * `FP32` Master Weights ($4\Psi$) + `FP32` Momentum $m$ ($4\Psi$) + `FP32` Variance $v$ ($4\Psi$): **$12\Psi$ bytes**
     * **Total static state before activations = $16\Psi$ bytes!** (A 7B model needs $16 \times 7 = 112$ GB just for weights/grads/optimizer—already bursting an 80 GB A100!).
  2. **Collective Communication Primitives & `Ring-AllReduce` Bandwidth Proof:**
     * `Broadcast`, `Reduce`, `AllGather`, `ReduceScatter`, and **`AllReduce = ReduceScatter + AllGather`**.
     * Derive on paper why naive Parameter-Server scales terribly as $O(N_{\text{GPUs}})$, whereas **`Ring-AllReduce`** across $P$ GPUs sends $2 \frac{P-1}{P} M$ bytes per GPU—**bandwidth-optimal and virtually independent of cluster size $P$**!
  3. **The 4D Parallelism Playbook:**
     * **Data Parallelism (`DDP`) $\to$ `ZeRO-1 / 2 / 3` (`FSDP`):** Sharding optimizer states (`ZeRO-1`: $4\Psi + \frac{12\Psi}{P}$), + gradients (`ZeRO-2`), + parameters (`ZeRO-3` / `PyTorch FSDP`: $\frac{16\Psi}{P}$ bytes per GPU, prefetching `AllGather` per layer right before compute!).
     * **Tensor Parallelism (`Megatron-LM`):** Splitting an MLP `ColumnParallelLinear` ($X [W_1 \mid W_2]$) followed by `RowParallelLinear` ($\begin{bmatrix}Y_1 & Y_2\end{bmatrix}\begin{bmatrix}V_1 \\ V_2\end{bmatrix} = Y_1 V_1 + Y_2 V_2$)—requiring **only ONE `AllReduce` per entire MLP or Multi-Head Attention block** (must stay within fast intra-node **NVLink**!).
     * **Pipeline Parallelism (`1F1B` Schedule):** Splitting layers across nodes; deriving the pipeline bubble fraction $\frac{P - 1}{M + P - 1}$ for $M$ micro-batches.
     * **Context Parallelism (`Ring Attention`):** Splitting a 1,000,000-token sequence across GPUs, passing $K, V$ blocks around a ring overlapped with FlashAttention online rescaling!
* **⚡ Practical Magic Lab (`distributed_labs/04_ring_allreduce_zeRO_and_tp_sim.py`):**
  * Use `torch.distributed` (`gloo` multi-process on local CPU or `nccl` on multi-GPU Kaggle T4x2!) to implement **`Ring-AllReduce` from raw `send`/`irecv` point-to-point primitives**, a miniature **`ZeRO-1` Sharded AdamW**, and **`Megatron` Column+Row Tensor-Parallel MLP**, verifying bit-identical gradients against single-device training!

---

### Module 6.5: High-Throughput Batched C++/CUDA RL Physics Engines (Game Engine $\times$ AI Synthesis!)
*Where your passion for Game Engine Dev becomes a superpower that 99% of ML researchers lack.*

* **Why Standard RL Is Broken:**
  * In standard `Gymnasium` Python RL, stepping 64 environments runs Python `env.step()` across OS pipes (`SubprocVecEnv`), serializing NumPy dicts, copying bytes to CPU RAM, then copying over PCIe to GPU for a tiny 2-layer policy forward pass, then copying actions back over PCIe to CPU! **95% of wall-clock time is wasted on IPC and PCIe transfers** (~5,000 steps/sec).
* **The Data-Oriented C++/CUDA Engine Architecture (`EnvPool` / `Madrona` / `IsaacGym` / `Warp` Style):**
  1. Store the state of **$N = 16{,}384$ independent game worlds** as a single **Data-Oriented `SoA` GPU Tensor struct** (`pos_x[16384]`, `pos_y[16384]`, `vel_x[16384]`, `rewards[16384]`, `dones[16384]`) permanently resident in GPU VRAM!
  2. One CUDA kernel (`step_worlds_kernel<<<64, 256>>>`) advances physics, raycasts/collisions, rewards, and **automatic in-kernel episode resets (`if (done) reset_world(env_id, rng)`)** for all 16,384 worlds in **< 0.05 milliseconds**!
  3. Zero-copy PyTorch tensor views wrap those exact VRAM pointers: `PPO` forward/backward runs on the GPU right next to the physics kernel—**zero Python loops, zero PCIe copies**, hitting **5,000,000+ environment steps per second** and training an agent in **8 seconds instead of 6 hours**!
* **⚡ Practical Magic Lab (`rl_physics_engine/05_batched_soa_env_prototype/`):**
  * Build a vectorized 2D physics game (e.g., **Continuous-Thrust Asteroids / Lunar Lander / 2D Raycast Tag**) first in naive OOP Python `Gym` (`~3,000 SPS`), then in C++20 `SoA` OpenMP / CUDA (`>2,000,000 SPS`), and watch `PPO` solve the game in real-time before your coffee cools!

---

## 📝 Required `notes/` Artifacts for Phase 06

1. `01_llm_memory_and_roofline_bible.md` — Your personal 1-page reference sheet deriving `16Ψ` training memory, activation memory, KV-cache bytes, Prefill vs Decode arithmetic intensity, and `ZeRO`/`TP`/`PP` communication volume formulas from memory!
2. `02_speculative_decoding_proof.md` — Clean probabilistic proof that speculative rejection sampling preserves the exact target distribution $p(x)$.
3. `03_gpu_rl_engine_soa_design.md` — Memory layout & auto-reset state-machine diagram for a zero-copy GPU-resident 16,384-world RL physics engine.

---

## 🏆 Phase 06 Boss Fight: Build a Real Systems Engine (`TinyInfer` OR `HyperEnv`)

Choose **one** (or both across consecutive quarters!) to engineer to rock-solid completion:

### Option A: `TinyInfer` — Zero-Python C++20/CUDA Quantized LLM Inference Engine
* Load real open-weight Transformer weights (e.g., SmolLM / Llama-3.2-1B / Qwen-0.5B exported via `mmap` binary format) **purely in C++20 + CUDA (zero Python, zero PyTorch at runtime!)**.
* Implement `RMSNorm`, `RoPE`, `GQA` with Paged/Contiguous **KV-Cache**, `SwiGLU`, and **Fused `INT8`/`INT4` Weight-Only GEMV**.
* Benchmark prefill tok/s, decode tok/s, and % of theoretical HBM bandwidth peak against PyTorch and `llama.cpp`!

### Option B: `HyperEnv` — 10,000-World GPU-Resident C++/CUDA RL Physics Engine
* Build a multi-entity 2D rigid-body/raycast game engine in C++20 + CUDA stepping $16{,}384$ worlds simultaneously in `SoA` VRAM buffers, exposed zero-copy via `Pybind11` / `DLPack` to PyTorch/JAX `PPO`.
* Include a lightweight real-time visualizer (or replay video exporter) watching your agent go from random flailing at $t = 0\text{s}$ to superhuman mastery at $t = 15\text{s}$ at **>2,000,000 FPS**!
