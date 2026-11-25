# 📜 The Manifesto, Old-School Mastery Protocol & Game Engine Rosetta Stone

---

## 1. Who I Am Training to Become

When someone asks what my end goal is, the exact title is:

> **"I am training to become an ML Systems Researcher & Research Engineer."**

And the two-sentence explanation:

> *"Most practitioners treat Python frameworks as black boxes to assemble standard models. My goal is to master both the mathematical why behind every modelling decision and the physical silicon underneath—using C++, compilers, and CUDA to invent hardware-aware statistical algorithms, deep learning engines, and ultra-fast RL simulation environments."*

### The "Unicorn Bridge"
The modern AI world suffers from a deep cultural split:
1. **Pure Math & Statistics Theorists:** Understand probability distributions, convergence proofs, and calculus deeply, but often write sequential, memory-bound Python code that hits a wall at scale.
2. **Pure Systems & Software Engineers:** Command caches, C++ memory layouts, and threads, but treat gradients, conditioning, variance propagation, and statistical inference as opaque formulas handed down from above.

**This curriculum trains me to be both in one brain.** Coming from a strong BSc in Statistics (3.99/5.00 CGPA) while deliberately rebuilding every mathematical, architectural, and hardware concept from zero prerequisites, I am targeting roles where these two worlds collide:
* **Research Engineer / Systems Scientist at a Frontier Lab** (Google DeepMind, OpenAI, Anthropic, Meta FAIR, NVIDIA Research): Translating new mathematical ideas into silicon-efficient kernels and scalable training systems.
* **RL & Simulation Systems Architect:** Designing data-oriented C++/CUDA physics and game environments where thousands of agents learn simultaneously at millions of frames per second.
* **Research-Intensive Graduate Researcher** (e.g., Mila, Vector Institute, Waterloo, UofT, ETH, CMU): Publishing at the intersection of statistical machine learning and hardware-software co-design.

---

## 2. The Old-School Mastery Protocol (Learning Without AI Crutches)

Surface-level familiarity evaporates under pressure; first-principles mastery compounds for decades. Because the goal of this multi-year journey is **real understanding—not hype, vibe-coding, or illusion of competence**, I follow strict rules of engagement:

### Rule 1: Zero Black Boxes & Zero Skipped Prerequisites
Even when a topic looks familiar from university statistics (e.g., variance, eigenvalues, maximum likelihood, chain rule), ask:
1. *What is the geometric picture?*
2. *Why was this exact formulation chosen over simpler alternatives?*
3. *What breaks numerically or physically on real hardware when dimensions hit $10^4$ or parameters hit $10^9$?*

### Rule 2: Pen-and-Paper Before Keyboard
Never implement a backward pass, normalization layer, probability loss, stride index formula, or tiled matrix loop by guessing shapes until errors disappear.
1. Take a blank sheet of paper (or tablet).
2. Write out a tiny concrete example ($2 \times 3$ matrix, 4-thread block, 3-node graph).
3. Derive the exact index/calculus math by hand.
4. **Only then** translate your derivation into code.

### Rule 3: Primary Sources First
Learn from the people who invented or built the systems:
* **Textbooks & University Lectures** over 5-minute blog summaries.
* **Official Documentation, Header Files & Hardware Manuals** (`cppreference.com`, PyTorch `ATen` source headers, NVIDIA CUDA C++ Programming Guide, PTX ISA docs) over copy-pasted snippets.
* **Original Seminal Papers** (read using Keshav's Three-Pass Method in [READING_LIST_AND_PAPERS.md](./READING_LIST_AND_PAPERS.md)) so you witness the exact problem the authors faced before their solution became textbook dogma.

### Rule 4: The "No-AI Crutch" Debugging Escalation Ladder
Do **not** use AI code generation, autocomplete (disable Copilot/inline completions in **VS Code**, **Visual Studio**, and **Google Colab** settings while studying!), or LLM explanations as a first resort. Struggle is the exact physical signal of neural myelination. When stuck on a bug or proof, climb this ladder in strict order:
1. **Minutes 0–30 (Inspect Physical Reality):** Print shapes, strides, dtypes, `isnan`/`isinf`, memory addresses, or small $2 \times 2$ toy inputs with `torch.manual_seed(0)`. Step through line-by-line with the **VS Code Python Debugger** (`pdb`), inspect raw bytes/registers/assembly in the **Visual Studio Native C++ Debugger** (*Memory* & *Disassembly* windows), or check small matrix identities live in **MATLAB**. Check finite-difference numerical gradients!
2. **Minutes 30–60 (Primary Docs & Tools):** Read the official function signature/docs, check `cppreference.com` or MATLAB docs, inspect generated `MSVC`/`NVCC` assembly on [Compiler Explorer (`godbolt.org`)](https://godbolt.org/), or re-read the textbook derivation.
3. **Minutes 60–90 (Step Away):** Take a 15-minute walk without screens and redraw the dataflow diagram from memory on paper. Half of all systems bugs die here.
4. **Minute 90+ (Targeted Unstuck Query Only):** If and *only* if still genuinely blocked after 90 minutes of documented effort, ask a **narrow, diagnostic question** (e.g., *"Here is my hand derivation of BatchNorm's `dgamma` and my finite-difference check fails on off-diagonal terms—which step of my chain rule has a missing sum?"*)—**never** *"Write this function for me."* Immediately close the chat and implement the fix from memory yourself.

### Rule 5: Theory Must Perform Practical Magic
Theory without practice becomes sterile; practice without theory becomes cargo-culting. Every single module in this repo pairs derivation with an immediate **visual, numerical, or performance experiment**:
* Derived SVD? Immediately compress an image and watch singular modes reconstruct faces.
* Derived Kaiming Init? Plot 50-layer activation histograms snapping from dead zeros to clean Gaussians.
* Learned CPU cache lines & SIMD? Watch a 100,000-particle 2D physics simulation jump from 15 FPS to 1,200 FPS in Visual Studio `Release` mode.
* Derived Online Softmax? Watch sequence length $N = 16,384$ run on Google Colab without `CUDA Out of Memory`.

### Rule 6: Strict Measurement & Ground-Truth Culture
1. **Correctness First:** Every custom NumPy, C++, Triton, or CUDA op must be verified against a trusted independent oracle (**MATLAB** Symbolic/Stats Toolboxes, finite differences, or PyTorch `float64` / `SymPy`) using `assert_close` before timing a single millisecond.
2. **Performance Second:** Never claim code is "fast." Report hardware specs, warmup protocol, latency (ms), effective memory bandwidth (GB/s), compute throughput (GFLOP/s or TFLOP/s), and % of theoretical Roofline peak.
3. **Log Everything:** End every session—whether 20 minutes or 5 hours—with a four-bullet entry in [JOURNAL.md](./JOURNAL.md).

---

## 3. The Game Engine $\leftrightarrow$ ML Systems Rosetta Stone

Loving **Game Engine Development** is a massive unfair advantage in **ML Systems Research**. Why? Because modern GPUs were literally born for 3D graphics engines, and a deep learning runtime *is* a specialized, differentiable game/physics engine!

Whenever you encounter a new ML Systems concept in Phases 02–06, map it directly to its twin in Game Engine Architecture:

| Game Engine Concept | ML Systems Twin | Why They Are Physically Identical |
| :--- | :--- | :--- |
| **Main Game Loop** (`Input -> StepPhysics(dt) -> Render()`) | **Training / RL Rollout Loop** (`DataLoader -> Forward/Backward -> Step(lr)`) | Both are tight frame-budget loops where any Python/CPU stall starves the GPU (`frame-time spike` = `GPU bubble`). |
| **Entity-Component-System (ECS) & Structure of Arrays (`SoA`)** | **Contiguous Tensors & Strided Storage** | Instead of `Particle {x,y,z,vx,vy,vz}` scattered on the heap (`AoS`), storing contiguous `float x[N], y[N], z[N]` (`SoA` / Tensor) fills 64-byte CPU cache lines and 128-byte GPU coalesced transactions with 100% useful math. |
| **Frame Arena / Linear Bump Allocator & Object Pools** | **PyTorch `CUDACachingAllocator` & Static Graph Memory Planning** | Calling OS `malloc`/`free` or `cudaMalloc`/`cudaFree` every frame/step takes microseconds to milliseconds and fragments RAM. Both engines grab one giant block upfront and hand out pointer offsets in $O(1)$ nanoseconds! |
| **3D Transform Matrices (`Mat4`), Quaternions & Change of Basis** | **Linear Layers (`xW`), Orthogonal Projections & `RoPE` Attention** | `RoPE` (Rotary Positional Embeddings) in LLMs literally rotates query/key vectors in 2D subspaces by angle $m\theta$—the exact 2D rotation matrix used in every 2D game engine! |
| **CPU SIMD Vector Math (`AVX2` / `NEON` `__m256`)** | **Vectorized CPU Tensor Kernels (`llama.cpp` `ggml` kernels)** | Processing 8 `float32` numbers in a single CPU clock instruction via Fused Multiply-Add (`FMA`). |
| **HLSL / GLSL / Vulkan Compute Shaders (`SV_DispatchThreadID`)** | **CUDA `__global__` Kernels (`blockIdx * blockDim + threadIdx`) & Triton** | Thousands of lightweight threads executing Single-Instruction-Multiple-Threads (SIMT) in 32-thread warps, sharing fast on-chip SRAM (`groupshared` in HLSL $\equiv$ `__shared__` in CUDA!). |
| **RenderGraph / Shader Fusion** (avoiding writing intermediate G-Buffers to VRAM) | **Compiler Operator Fusion** (`torch.compile` / XLA / `FlashAttention`) | Reading/writing VRAM (HBM) costs ~100x more energy and time than doing ALU math in registers/SRAM. Keeping intermediate values on-chip between passes/ops is the #1 optimization in both fields. |
| **Tracy / RenderDoc / PIX Frame Profilers** | **`torch.profiler` / Perfetto / NVIDIA Nsight Systems & Compute** | Visual timelines showing CPU command-queue submission vs asynchronous GPU hardware execution streams. |
| **Rigid-Body / Particle Physics Engines (`Box2D`, `PhysX`, `MuJoCo`)** | **Batched C++/CUDA RL Environments (`EnvPool`, `Madrona`, `Warp`)** | Stepping 10,000 game worlds in parallel inside GPU VRAM so an RL policy trains directly on GPU tensors without ever touching PCIe or Python! |

---

## 4. Definition of "Done" for Any Module

You may check off `- [x]` on a module in [README.md](./README.md) **only** when all four conditions hold:
1. **Feynman Test:** You wrote a short, self-contained note in that phase's `notes/` folder explaining *why* the concept exists and deriving its core equation without looking at the book.
2. **From-Scratch Code:** You wrote, ran, and verified the module's hands-on lab in `labs/` or `src/`.
3. **Visual / Metric Proof:** You generated the plot, numerical check, or latency benchmark proving theory worked in practice.
4. **Journal Logged:** You recorded the struggle and insight in [JOURNAL.md](./JOURNAL.md).
