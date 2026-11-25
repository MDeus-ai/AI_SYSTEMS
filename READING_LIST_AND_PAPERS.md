# 📚 The Complete Chronological Reading List & Seminal Papers Syllabus

> **Rule of Thumb:** Do **not** hoard PDFs or try to read everything at once. Read **just-in-time** strictly according to the Phase you are actively executing. Primary textbooks, university lectures, and original papers beat hype threads every single time.

---

## 🧭 How to Read an ML Systems Paper (Adapted from S. Keshav's *Three-Pass Method*)

Never read a paper linearly from page 1 to page 12 on the first try—you will bog down in notation before grasping *why* the paper exists. Use three deliberate passes and record notes using [PAPER_TEMPLATE.md](./99_Verification_and_Papers/papers_notes/PAPER_TEMPLATE.md):

1. **Pass 1 — The 10-Minute Radar Scan:**
   * Read **Title, Abstract, Introduction**, section headings, **Roofline/Architecture Figures**, and **Conclusion**.
   * Answer in 3 sentences: *What exact mathematical or hardware bottleneck broke prior work? What is the single core trick? What speedup/accuracy did it unlock?*
2. **Pass 2 — The 60-Minute Mechanism Pass:**
   * Read the body carefully, annotating every figure and algorithm box, while temporarily skipping long appendix proofs.
   * Identify whether the bottleneck solved was **Compute-Bound**, **Memory-Bandwidth-Bound**, **Memory-Capacity-Bound**, or **Communication/Overhead-Bound**.
3. **Pass 3 — The Re-Invention Pass (Reserved for Seminal Classics):**
   * Close the PDF. Take a blank sheet of paper. Re-derive the core equation or SRAM tiling loop **from memory** as if *you* were inventing it in the lab, then implement a minimal working reproduction in code!

---

## 📖 Core Textbooks & Open Courses (Mapped by Phase)

### Phase 00: Math, Statistics, Floats & Scalar Autograd
* **Visual Series:** 3Blue1Brown — [Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) & [Essence of Calculus](https://www.3blue1brown.com/topics/calculus)
* **Free Textbook:** Deisenroth, Faisal & Ong — [*Mathematics for Machine Learning*](https://mml-book.github.io/) *(Ch. 2–7)*
* **Course:** Gilbert Strang — [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) & [MIT 18.065 Matrix Methods in ML](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)
* **Monograph:** Terence Parr & Jeremy Howard — [*The Matrix Calculus You Need For Deep Learning*](https://explained.ai/matrix-calculus/)
* **Probability & Info Theory:** Joe Blitzstein — [Harvard Stat 110](https://projects.iq.harvard.edu/stat110/home), [Seeing Theory](https://seeing-theory.brown.edu/), and Chris Olah's [*Visual Information Theory*](https://colah.github.io/posts/2015-09-Visual-Information/)
* **Computer Arithmetic:** David Goldberg (1991) — *"What Every Computer Scientist Should Know About Floating-Point Arithmetic"* + [float.exposed](https://float.exposed/)

### Phase 01: Deep Learning Architectures & Decisions (Pure NumPy)
* **Free Textbook (Primary):** Simon J.D. Prince — [*Understanding Deep Learning* (MIT Press)](https://udlbook.github.io/udlbook/)
* **Free Interactive Book:** Zhang, Lipton, Li & Smola — [*Dive into Deep Learning* (`d2l.ai`)](https://d2l.ai/)
* **Lectures:** Andrej Karpathy — [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html), [Stanford CS231n](https://cs231n.stanford.edu/), and [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
* **Reinforcement Learning:** OpenAI — [*Spinning Up in Deep RL*](https://spinningup.openai.com/) & Sutton & Barto — [*Reinforcement Learning: An Introduction (2nd Ed)*](http://incompleteideas.net/book/the-book-2nd.html)

### Phase 02: PyTorch & JAX Internals
* **Internals Deep Dives:** Edward Z. Yang — [*PyTorch Internals*](http://blog.ezyang.com/2019/05/pytorch-internals/) & Horace He — [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html)
* **Books & Docs:** Eli Stevens et al. — *Deep Learning with PyTorch*, Grigory Sapunov — *Deep Learning with JAX*, and [Official JAX Docs](https://docs.jax.dev/en/latest/)

### Phase 03: Modern C++20, Hardware Architecture & Game-Engine DOD
* **C++ References:** [LearnCpp.com](https://www.learncpp.com/), Bjarne Stroustrup — *A Tour of C++ (3rd Ed)*, and [cppreference.com](https://en.cppreference.com/)
* **Hardware Bible:** Bryant & O'Hallaron — *Computer Systems: A Programmer's Perspective (`CS:APP`, 3rd Ed)* *(Ch. 1, 3, 5, 6, 9)*
* **Free HPC Book:** Sergey Slotin — [*Algorithmica: High-Performance Computing*](https://en.algorithmica.org/hpc/)
* **Game Engine & DOD:** Jason Gregory — *Game Engine Architecture (3rd Ed)*, Casey Muratori — [*Performance-Aware Programming*](https://www.computerenhance.com/), and Mike Acton's CppCon 2014 *"Data-Oriented Design and C++"*

### Phase 04: The Bridge & Custom DL Framework
* **Primary Course:** Zico Kolter & Tianqi Chen — [CMU 10-414 / 10-714: *Deep Learning Systems* (`dlsyscourse.org`)](https://dlsyscourse.org/)
* **Docs:** [Pybind11 Docs](https://pybind11.readthedocs.io/en/stable/) & [PyTorch C++ Extensions Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)

### Phase 05: GPU Architecture, Triton & Raw CUDA
* **GPU Bible:** Kirk, Hwu & El Hajj — *Programming Massively Parallel Processors (`PMPP`, 4th Ed)*
* **Interactive Puzzles:** Sasha Rush — [Triton Puzzles](https://github.com/srush/Triton-Puzzles) & [GPU Puzzles](https://github.com/srush/GPU-Puzzles)
* **Practitioner Gold:** Simon Boehm — [*CUDA Matmul Optimization Worklog*](https://siboehm.com/articles/22/CUDA-MMM), [GPU MODE Lectures](https://github.com/gpu-mode/lectures), [OpenAI Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html), and [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Phase 06: Compilers, Quantization, Serving, Distributed Scale & RL Engines
* **ML Compilers:** Tianqi Chen — [*Machine Learning Compilation* (`mlc.ai`)](https://mlc.ai/)
* **Scaling Bibles:** Google DeepMind — [*How To Scale Your Model*](https://jax-ml.github.io/scaling-book/) & Hugging Face — [*The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
* **Masterwork Codebases:** [`karpathy/llm.c`](https://github.com/karpathy/llm.c), [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp), [`vwxyzjn/cleanrl`](https://github.com/vwxyzjn/cleanrl), and [`madrona-engine`](https://madrona-engine.github.io/)

---

## 📜 The 30 Seminal Papers (Chronological Study Order)

### Group I — Mathematical & Architectural Decisions (Read in Phase 01)
1. **Xavier Init:** Glorot & Bengio (2010) — [*Understanding the difficulty of training deep feedforward neural networks*](https://proceedings.mlr.press/v9/glorot10a.html)
2. **Kaiming Init:** He et al. (2015) — [*Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*](https://arxiv.org/abs/1502.01852)
3. **ResNets:** He et al. (2015) — [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385)
4. **BatchNorm:** Ioffe & Szegedy (2015) — [*Batch Normalization: Accelerating Deep Network Training*](https://arxiv.org/abs/1502.03167)
5. **LayerNorm:** Ba, Kiros & Hinton (2016) — [*Layer Normalization*](https://arxiv.org/abs/1607.06450)
6. **RMSNorm:** Zhang & Sennrich (2019) — [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467)
7. **GLU / SwiGLU:** Shazeer (2020) — [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202)
8. **Adam:** Kingma & Ba (2014) — [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980)
9. **AdamW:** Loshchilov & Hutter (2017/2019) — [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101)
10. **Transformers:** Vaswani et al. (2017) — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
11. **RoPE:** Su et al. (2021) — [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864)
12. **Grouped-Query Attention (`GQA`):** Ainslie et al. (2023) — [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245)
13. **VAEs:** Kingma & Welling (2013) — [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114)
14. **Diffusion (`DDPM`):** Ho, Jain & Abbeel (2020) — [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239)
15. **PPO:** Schulman et al. (2017) — [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347)

### Group II — Frameworks & Automatic Differentiation (Read in Phases 02 & 04)
16. **AD Survey:** Baydin et al. (2018) — [*Automatic Differentiation in Machine Learning: a Survey*](https://arxiv.org/abs/1502.05767)
17. **PyTorch 1 Design:** Paszke et al. (NeurIPS 2019) — [*PyTorch: An Imperative Style, High-Performance Deep Learning Library*](https://arxiv.org/abs/1912.01703)
18. **JAX Tracing Design:** Frostig, Johnson & Leary (SysML 2018) — [*Compiling machine learning programs via high-level tracing*](https://mlsys.org/Conferences/doc/2018/146.pdf)

### Group III — GPU Kernels & Hardware-Aware Algorithms (Read in Phase 05)
19. **Triton Compiler:** Tillet, Kung & Cox (MAPL 2019) — [*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
20. **Online Softmax:** Milakov & Gimelshein (2018) — [*Online normalizer calculation for softmax*](https://arxiv.org/abs/1805.02867)
21. **FlashAttention:** Dao et al. (NeurIPS 2022) — [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135)
22. **FlashAttention-2:** Dao (2023) — [*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691)
23. **Mamba (Hardware-Aware Parallel Scan):** Gu & Dao (2023) — [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752)

### Group IV — Compilers, Quantization, Serving, Distributed & RL Systems (Read in Phases 06 & 07)
24. **PyTorch 2 Compiler:** Ansel et al. (ASPLOS 2024) — [*PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation*](https://pytorch.org/assets/pytorch2-2.pdf)
25. **Mixed Precision:** Micikevicius et al. (ICLR 2018) — [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740)
26. **Outlier Quantization (`LLM.int8()` / `AWQ`):** Lin et al. (MLSys 2024) — [*AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*](https://arxiv.org/abs/2306.00978)
27. **PagedAttention (`vLLM`):** Kwon et al. (SOSP 2023) — [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180)
28. **Speculative Decoding:** Leviathan, Kalman & Matias (ICML 2023) — [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192)
29. **ZeRO & Megatron-LM:** Rajbhandari et al. (SC 2020) — [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054) & Shoeybi et al. (2019) — [*Megatron-LM*](https://arxiv.org/abs/1909.08053)
30. **GPU-Resident RL Simulation (`Madrona` / `EnvPool`):** Shacklett et al. (SIGGRAPH 2023) — [*An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation*](https://arxiv.org/abs/2305.11286)
