# ⚙️ Phase 02: Mastery of PyTorch & JAX Internals (Deconstructing the Front-End)

> **Prerequisites:** [Phase 00](../00_Math_and_Numeric_Foundations/README.md) & [Phase 01](../01_Deep_Learning_From_Scratch/README.md) completed. *(Note: You already completed Modules 2.1 and 2.2 and started Module 2.3 in your preserved notebooks below!)*  
> **Estimated Pace:** 3–4 Months.  
> **Tools Used in Phase 02:** **VS Code** (Python + Jupyter extension for local tensor/autograd dissection) + **Google Colab** (GPU runtime for benchmarking `torch.cuda.synchronize()`, `torch.compile`, `@jax.jit` XLA HLO fusion, and parallel HMC chains) + **Perfetto UI** + **MATLAB** (posterior & numerical verification).  
> **Big Idea (1 Sentence):** *Unlearn high-level `torch.nn` / `flax` wrappers and master the two competing philosophies of modern ML systems: PyTorch's imperative dynamic C++ dispatcher (`Storage` + `Autograd` DAG) and JAX's pure functional compiler pipeline (`PyTrees` + `jaxpr` + `XLA HLO`).*

---

## 📂 Preserved Workspace Notebooks (Your Head Start!)

* ✅ **[Module 1 (Tensor Memory Layout).ipynb](./Module%201%20%28Tensor%20Memory%20Layout%29.ipynb)** — Completed (`Storage`, `offset`, `stride()`, `is_contiguous()`, zero-copy transposes).
* ✅ **[Module 2 (Math Operations & Early Measurement Culture).ipynb](./Module%202%20%28Math%20Operations%20%26%20Early%20Measurement%20Culture%29.ipynb)** — Completed (`measure_speed` warmup + synchronization timer, in-place vs out-of-place, zero-stride broadcasting).
* 🔄 **[Module 3 (Autograd & Calculus Engines).ipynb](./Module%203%20%28Autograd%20%26%20Calculus%20Engines%29.ipynb)** — In Progress (`grad_fn`, `is_leaf`, `next_functions`).

---

## 🎮 The Game Engine & Systems Connection
* **PyTorch is like an Immediate-Mode Engine / Scripting Layer:** Every Python line (`y = x * w`) immediately dispatches into a compiled C++ kernel via a virtual lookup table (**The Dispatcher**), recording a trail of breadcrumbs (`AutogradMeta`) on the heap so `.backward()` can replay calculus in reverse.
* **JAX is like a Shader Graph Compiler:** Instead of executing operations one by one, `@jax.jit` traces your Python function once with abstract shaped "ghost" arrays, builds an entire static Intermediate Representation (`jaxpr` $\to$ `StableHLO`), fuses elementwise loops together so intermediate arrays never touch RAM, and compiles straight to native machine/GPU instructions via **XLA**!

> **🛠️ Phase 02 Tooling Setup:** Run CPU storage, stride, and autograd graph inspections locally in **VS Code** (`.ipynb` / `.py`), cross-check HMC posterior statistics in **MATLAB**, and switch to **Google Colab (`T4` GPU runtime)** whenever running GPU synchronization benchmarks, `torch.compile`, `@jax.jit` XLA HLO fusion dumps, and the Boss Fight HMC GPU speedup showdown!

---

## 📚 Primary Learning Resources for Phase 02

* **Foundational Internals Essays & Books:**
  * Edward Z. Yang — [*PyTorch Internals*](http://blog.ezyang.com/2019/05/pytorch-internals/) *(Mandatory reading! Every ML systems engineer knows this post by heart.)*
  * Horace He — [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html) *(Compute-bound vs Memory-bandwidth-bound vs Overhead-bound!)*
  * Eli Stevens, Luca Antiga, Thomas Viehmann — *Deep Learning with PyTorch* (Chapters 3, 5, 14)
  * Official [PyTorch Autograd Mechanics Documentation](https://pytorch.org/docs/stable/notes/autograd.html)
  * Official [JAX Documentation: Thinking in JAX, Autodiff Cookbook & Sharp Bits](https://docs.jax.dev/en/latest/)
  * Google DeepMind (Austin et al.) — [*How To Scale Your Model* (`jax-ml/scaling-book`, Chapters 1–3 on Roofline & Transformer Math)](https://jax-ml.github.io/scaling-book/)

---

## 🧱 Chronological Modules

### Module 2.1: Tensor Memory Layout — Storage, Views, Strides & Aliasing `[COMPLETED ✅]`
*Before you manipulate raw pointers in C++ (Phase 03), see how Python hides 1D physical RAM behind metadata.*

* **Core Mastery Checklist (Review & Solidify):**
  1. Logical Tensor (`TensorImpl`: shape, stride, storage_offset, dtype, device) vs Physical Memory (`Storage`: flat 1D buffer of bytes).
  2. Exact physical 1D offset formula for an $N$-dimensional tensor:
     $$\text{storage\_index}(i_0, i_1, \dots, i_{N-1}) = \text{offset} + \sum_{k=0}^{N-1} i_k \cdot \text{stride}[k]$$
  3. Why `.view()` fails on non-contiguous tensors (`points.t().view(-1)` raises `RuntimeError`!) whereas `.reshape()` silently allocates a copy when non-contiguous, and `.contiguous()` materializes physical row-major order.
  4. **Advanced Stride Magic (`torch.as_strided`):** How sliding convolution windows and overlapping sequence chunks can be created in $O(1)$ time with **zero memory allocation** purely by crafting custom strides—and why mutating overlapping strided views is dangerous!

---

### Module 2.2: Allocation Semantics, Broadcasting & Measurement Culture `[COMPLETED ✅]`
*Why Systems Engineers never trust wall-clock intuition without warmups, hardware synchronization, and memory accounting.*

* **Core Mastery Checklist (Review & Extend):**
  1. Out-of-place (`y = x * 2` $\to$ new heap/VRAM buffer) vs In-place (`x.mul_(2)` $\to$ mutates storage; why Autograd throws errors if an in-place op overwrites an activation saved for `.backward()`!).
  2. Right-to-left broadcasting rules and the **zero-stride virtual expansion trick** (`stride = (0, 1)`).
  3. **Upgrade Exercise (`labs/02_overhead_vs_bandwidth_vs_compute.py`):** Read Horace He's *Go Brrrr* article, then use your `measure_speed` timer to plot latency across tensor sizes from $N = 10$ to $N = 10^8$ for (a) tiny ops (`x + 1` — **Python/dispatch overhead bound**), (b) massive elementwise ops (`torch.sin(x)` — **DRAM bandwidth bound**), and (c) large matrix multiply (`x @ x` — **FLOP compute bound**).

---

### Module 2.3: PyTorch's Autograd Engine & The C++ Dispatcher `[IN PROGRESS 🔄]`
*Completing your `Module 3` notebook: how PyTorch tracks calculus dynamically and routes calls to CPU/CUDA/Autograd backends.*

* **Concepts Built From Scratch:**
  1. **Anatomy of the Dynamic DAG:** Leaf tensors (`is_leaf=True`, user-created weights where `.grad` accumulates) vs intermediate tensors (`grad_fn=<MulBackward0>`), traversing `grad_fn.next_functions` all the way to `AccumulateGrad`.
  2. **Why Gradients Accumulate (`+=`) by Default:** Why forgetting `optimizer.zero_grad(set_to_none=True)` adds step $t$'s gradient onto step $t-1$'s gradient—and why `set_to_none=True` is faster than zeroing memory (`memset` costs memory bandwidth; setting pointer to `nullptr` is free!). Also why gradient accumulation lets a single laptop GPU simulate massive batch sizes!
  3. **Saved Tensors & The Activation Memory Footprint:** Inspect `y.grad_fn._saved_self`! Measure exact RAM consumed during forward pass *with* `requires_grad=True` vs `with torch.no_grad():` vs `with torch.inference_mode():` (*Why is `inference_mode` even faster than `no_grad`?* Because it completely disables version-counter & view-tracking metadata bumps in C++!).
  4. **Custom `torch.autograd.Function`:** Writing static `forward(ctx, ...)` (`ctx.save_for_backward(...)`) and `backward(ctx, grad_output)` + verifying with `torch.autograd.gradcheck` in `float64`. *(This is the exact Python class that will wrap your C++ and CUDA kernels in Phases 04 and 05!)*
  5. **The PyTorch Dispatcher Mental Model:** How `torch.add(a, b)` inspects a 64-bit `DispatchKeySet` bitmask on `a` and `b` (`AutogradCPU`, `CPU`, `CUDA`, `Batched`, `Tracer`) to jump to the exact right C++ kernel without `if/else` spaghetti.
* **⚡ Practical Magic Lab (Finish `Module 3` Notebook):**
  * Write a Python recursive graph walker that prints the entire `grad_fn.next_functions` tree of any equation.
  * Implement a custom `SwiGLUFunction(torch.autograd.Function)` with manual `ctx.save_for_backward`, pass `torch.autograd.gradcheck`, and compare peak memory against naive eager `SwiGLU`.

---

### Module 2.4: Building `nanoGPT` & KV-Cached Inference in Raw Tensor PyTorch
*Proving you don't need `torch.nn` black boxes to build modern LLMs.*

* **Concepts Built From Scratch:**
  1. Implementing a complete modern Decoder-Only Transformer (`RMSNorm`, `RoPE`, `Grouped-Query Attention`, `SwiGLU`) using **raw `torch.Tensor` parameters, `torch.einsum` / `matmul`, and manual `AdamW`** (no `nn.Linear`, no `nn.MultiheadAttention`, no `torch.optim`!).
  2. **Implementing a `KVCache` from Scratch:**
     * *Without KV-Cache:* Generating token $t$ recomputes $Q, K, V$ for all previous $t-1$ tokens—making generating $T$ tokens cost **$O(T^3)$** work!
     * *With Pre-allocated Contiguous KV-Cache:* Pre-allocate `k_cache, v_cache = torch.zeros(B, H_kv, max_seq_len, D)` once, slice-update position `[:, :, pos:pos+1, :] = k_new`, and watch generation latency drop by **10x–50x**!
* **⚡ Practical Magic Lab (`labs/04_raw_tensor_nanogpt_kvcache.py`):**
  * Benchmark tokens/second generating 512 tokens *without* KV-cache vs *with* dynamic `torch.cat` KV-cache (bad: allocates every step!) vs *with* pre-allocated slice-indexed KV-cache (systems-grade!). Plot the latency-per-token curve across token index $1 \to 512$!

---

### Module 2.5: The JAX Paradigm — Pure Functions, PyTrees & Parallel Splittable PRNGs
*Why compiler architects designed JAX to forbid in-place mutation and global random seeds.*

* **Concepts Built From Scratch:**
  1. **Stateful OOP (PyTorch) vs Pure Functional (JAX):** Why side effects (mutating globals or `x[0] = 5` in-place) make compiler reordering and automatic parallelization unsafe, and why JAX uses immutable functional updates (`x = x.at[0].set(5)`—which XLA compiles back into a zero-copy in-place mutation on hardware when safe!).
  2. **PyTrees (`jax.tree.map`):** Representing arbitrary nested dictionaries/tuples of model weights cleanly so one line (`jax.tree.map(lambda w, g: w - lr * g, params, grads)`) updates an entire 100-layer Transformer!
  3. **🔍 Why Did JAX Reject `manual_seed()` for Explicit Splittable PRNG Keys?**
     * Stateful RNGs (`np.random.randn()`) mutate a hidden sequential counter: if two GPU kernels or compiler-reordered branches call `randn()`, execution order changes the random numbers (non-reproducible race conditions!).
     * JAX uses counter-based cryptographic hashing (**Threefry**): `key, subkey1, subkey2 = jax.random.split(key, 3)` is a pure deterministic function that parallelizes across 10,000 GPU cores or RL environments reproducibly!
* **⚡ Practical Magic Lab (`labs/05_jax_pytrees_and_prng.py`):**
  * Implement your Phase 01 MLP as a pure functional PyTree dictionary of arrays, initialize weights cleanly by splitting PRNG keys across layers, and verify bit-exact reproducibility regardless of function call order.

---

### Module 2.6: Composable Transformations (`grad`, `jvp`, `vjp`, `vmap`, `lax.scan`, `jit`) & Reading XLA IR
*Seeing a machine learning compiler optimize your math in real time.*

* **Concepts Built From Scratch:**
  1. **`jax.grad` & `jax.value_and_grad`:** Differentiating Python functions directly—including higher-order derivatives `grad(grad(f))`!
  2. **The Pearlmutter Trick (`Hessian-Vector Products` via Forward-over-Reverse `jvp` of `grad`):** Remember in Phase 00 that storing an $N \times N$ Hessian takes Exabytes? Prove and code how `jvp(grad(loss), (params,), (v,))` computes the **exact** product $H v \in \mathbb{R}^N$ in **$O(N)$ memory and ~2x the cost of a single gradient** without ever materializing $H$! *(Secret weapon for Phase 07 Second-Order Optimizers!)*
  3. **`jax.vmap` (Automatic Vectorization):** Write logic for **one single sample** (no batch dimension bugs!), wrap in `jax.vmap`, and let JAX push the batch axis down through every primitive op!
  4. **`jax.lax.scan` (Compiled Sequential Loops):** Why a Python `for t in range(1000):` inside `@jax.jit` unrolls 1,000 copies of the graph (horrible compile time!), whereas `lax.scan` compiles a tight native hardware `while` loop—making MCMC samplers, RNNs, and RL rollouts fly!
  5. **`@jax.jit` Tracing Pitfalls & Reading XLA HLO:** Dynamic shapes causing re-tracing overhead; printing `jax.make_jaxpr(fn)(x)` and `jit_fn.lower(x).compile().as_text()` to **literally see XLA fuse 5 separate Python array operations into 1 single `fusion` kernel**!
* **⚡ Practical Magic Lab (`labs/06_xla_fusion_and_hvp.py`):**
  * Write a 6-op activation function (`Swish` / `GELU` approximation). Inspect its uncompiled `jaxpr` (6 separate kernels $\to$ 6 RAM round-trips) vs its compiled XLA HLO text (1 fused loop!), and benchmark Eager PyTorch vs `torch.compile` vs `@jax.jit`!

---

## 📝 Required `notes/` Artifacts for Phase 02

1. `01_pytorch_storage_and_dispatcher.md` — Memory diagram separating `Variable`/`TensorImpl` metadata from `Storage` + summary of Edward Yang's Dispatcher keys.
2. `02_overhead_bandwidth_compute_regimes.md` — Synthesis of Horace He's 3 performance regimes with your own measured crossover curves.
3. `03_pytorch_vs_jax_compiler_mental_model.md` — Two-column comparison of State, RNG, Autodiff (`VJP` only vs composable `JVP`+`VJP`), Loops, and Compilation (`TorchDynamo` vs `jaxpr`/XLA).

---

## 🏆 Phase 02 Boss Fight: The Statistical & Deep Learning Dual-Engine Showdown

**Location:** `02_PyTorch_and_JAX_Internals/boss_fight/`

**Rules of Engagement:**
* Zero `torch.nn`, zero `torch.optim`, zero `flax`/`optax`. Raw tensors, PyTrees, and calculus primitives only!

**Deliverables:**
1. **Challenge A (Raw-Tensor MLP / Mini-Transformer):**
   * Train the exact same architecture from identical initial weights in (1) Raw Eager PyTorch (`with torch.no_grad():` manual updates) and (2) Pure Functional `@jax.jit` + `jax.vmap` + `jax.value_and_grad`, proving identical loss curves and comparing epoch wall-clock speed.
2. **Challenge B (Your Statistics Flex — Vectorized Hamiltonian Monte Carlo `HMC`):**
   * Implement **Hamiltonian Monte Carlo** (using leapfrog physics integration driven by automatic log-posterior gradients $\nabla_\theta \log p(\theta \mid \mathcal{D})$!) to sample from a tricky banana-shaped Rosenbrock 2D/20D Bayesian posterior:
     * Version 1: Standard PyTorch Python `for` loop across 100 parallel chains $\times$ 20 leapfrog steps.
     * Version 2: JAX `@jax.jit` + `jax.vmap` (across 1,000 parallel chains!) + `jax.lax.scan` (across leapfrog steps!).
   * Measure samples-per-second and Effective Sample Size (`ESS`/sec). *Watching JAX `vmap`+`scan` sample 1,000 physics-guided Markov chains 50x–100x faster than Python loops will permanently wire into your brain why hardware-compiled systems change what statistical math is possible!*
