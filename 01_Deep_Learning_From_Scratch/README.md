# 🧬 Phase 01: Deep Learning Architectures, Decisions & Optimization Dynamics (Pure NumPy)

> **Prerequisites:** [Phase 00](../00_Math_and_Numeric_Foundations/README.md) completed (matrix geometry, Jacobians/VJPs, MLE/KL divergence, floating-point stability, and `MicroEngine`).  
> **Estimated Pace:** 4–6 Months.  
> **Tools Used in Phase 01:** **VS Code** (pure `numpy` + `matplotlib` / `TensorBoard` for 50-layer gradient histograms) + **MATLAB** (ground-truth verification of custom normalization Jacobians, probability distributions, and optimization trajectories).  
> **Big Idea (1 Sentence):** *Modern Deep Learning architectures are not arbitrary alchemy—every single component (`Kaiming Init`, `RMSNorm`, `SwiGLU`, `ResNets`, `AdamW`, `RoPE`, `GQA`, `VAE`, `PPO`) was engineered to solve a specific mathematical pathology in signal variance, gradient flow, conditioning, or memory complexity.*

---

## 🎯 Why Pure NumPy Before PyTorch/JAX?
In `MicroEngine` (Phase 00), every node was a single scalar `float`. Real neural networks operate on multidimensional arrays (**Tensors**). If you jump straight to `loss.backward()` in PyTorch, you never have to work out how batch dimensions, transposes, broadcasting backward reductions, normalization Jacobians, or attention masks actually differentiate in matrix form!

In Phase 01, **PyTorch and JAX are strictly forbidden** (except once at the very end of a test file if you want a secondary check alongside finite differences). You will write every forward pass, every analytical matrix backward pass, every initializer, every normalizer, every optimizer, and every architecture **in pure `numpy` inside VS Code**, using **MATLAB** (Symbolic Math & Statistics Toolboxes) whenever you want an independent oracle to verify tricky matrix Jacobians (`BatchNorm`, `RoPE`) or statistical distributions—unlocking complete x-ray vision into *why* every modern design decision exists.

---

## 📚 Primary Learning Resources for Phase 01

* **Core Textbook (Free Official Book — Gold Standard for "Why Decisions Are Made"):**
  * Simon J.D. Prince — [*Understanding Deep Learning* (MIT Press, 2023)](https://udlbook.github.io/udlbook/) *(Read Chapters 3–13, 15, 17–19 alongside these modules!)*
* **Hands-On Lecture Series:**
  * Andrej Karpathy — [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) *(Lectures 2–7: `makemore` MLP, Activations/Gradients/BatchNorm diagnostics, Becoming a Backprop Ninja, WaveNet, `nanoGPT`, Tokenization/BPE)*
  * Stanford [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/) *(Notes on Vectorized Backprop, CNNs, Weight Init, Optimization)*
  * Stanford [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) *(Lectures 1–4 on modern LLM architecture choices: RMSNorm, SwiGLU, RoPE, GQA, AdamW)*
  * Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola — [*Dive into Deep Learning*](https://d2l.ai/)
  * OpenAI — [*Spinning Up in Deep RL*](https://spinningup.openai.com/) *(Part 1 & Part 2 for Module 1.7)*
* **Seminal "Why This Decision Was Made" Papers (Read in Phase 01):**
  1. Glorot & Bengio (2010) — *Understanding the difficulty of training deep feedforward neural networks* (Xavier Init)
  2. He et al. (2015) — *Delving Deep into Rectifiers* (Kaiming Init) & *Deep Residual Learning for Image Recognition* (ResNets)
  3. Ioffe & Szegedy (2015) / Ba et al. (2016) / Zhang & Sennrich (2019) — *BatchNorm* $\to$ *LayerNorm* $\to$ *RMSNorm*
  4. Loshchilov & Hutter (2019) — *Decoupled Weight Decay Regularization* (`AdamW`)
  5. Vaswani et al. (2017) — *Attention Is All You Need* + Su et al. (2021) — *RoFormer: Enhanced Transformer with Rotary Position Embedding (`RoPE`)*

---

## 🧱 Chronological Modules

### Module 1.1: Vectorized Matrix Backpropagation & `einsum` Mastery
*Graduating from scalar derivatives to lightning-fast array calculus without ever memorizing brittle matrix-derivative tables.*

* **Concepts Built From Scratch:**
  1. **Shape-Driven + Index-Driven Derivative Method:** For $Z = X W + b$ where $X \in \mathbb{R}^{B \times D_{\text{in}}}$, $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, $b \in \mathbb{R}^{D_{\text{out}}}$, given upstream gradient $\bar{Z} = \frac{\partial L}{\partial Z} \in \mathbb{R}^{B \times D_{\text{out}}}$:
     * Write scalar index math first: $Z_{b, j} = \sum_i X_{b, i} W_{i, j} + b_j$.
     * Differentiate one scalar entry: $\frac{\partial L}{\partial W_{i, j}} = \sum_b \frac{\partial L}{\partial Z_{b, j}} X_{b, i} = (X^\top \bar{Z})_{i, j}$!
     * **Broadcast Backward Rule:** Whenever an operation *broadcasts* (stretches a dimension of size 1 to size $B$ in forward), its backward pass **sums** across that exact axis ($\bar{b} = \sum_{b=1}^B \bar{Z}_{b, :}$)! Why? Because broadcasting means the same parameter fed into $B$ computational branches (fan-out sum rule from Phase 00!).
  2. **`np.einsum` (Einstein Summation Notation):** Expressing matrix multiply (`"bi,ij->bj"`), batched matrix multiply (`"bti,bij->btj"`), trace, transpose, outer product, and multi-head attention scores in self-documenting index notation.
* **⚡ Practical Magic Lab (`labs/01_matrix_backprop_ninja.py`):**
  * Hand-derive and code forward + backward passes for `Linear`, `BroadcastAdd`, `ElementwiseMul`, `Sum`, `Max`, and fused `SoftmaxCrossEntropy` (proving on paper why the complicated Jacobian of Softmax + log-derivative of Cross-Entropy collapses into the shockingly clean $\bar{z} = p - y$!). Verify every single tensor gradient against finite differences to $10^{-7}$.

---

### Module 1.2: Signal Propagation, Weight Initialization & Activation Evolution
*Why did 10-layer neural networks completely fail to train in the 1990s—and how does elementary Statistics fix them in 4 lines of variance algebra?*

* **Concepts Built From Scratch & Statistical Proofs:**
  1. **The Symmetry Trap:** Why initializing $W = 0$ (or any constant) forces every neuron in a layer to compute identical outputs and receive identical gradients forever (rank-1 collapse).
  2. **Signal Variance Explosion & Collapse (Your Stats Superpower!):**
     Suppose input $x \in \mathbb{R}^{d_{\text{in}}}$ has independent zero-mean components with $\text{Var}(x_i) = 1$, and weights $W_{i,j} \sim \mathcal{N}(0, \sigma_w^2)$ independently. What is the variance of pre-activation $z_j = \sum_{i=1}^{d_{\text{in}}} x_i W_{i,j}$?
     $$\text{Var}(z_j) = \sum_{i=1}^{d_{\text{in}}} \text{Var}(x_i) \text{Var}(W_{i,j}) = d_{\text{in}} \sigma_w^2$$
     * If $\sigma_w = 1$ and $d_{\text{in}} = 1024$, after just 1 layer $\text{Var}(z) = 1024$; after 10 layers $\text{Var}(z) = 1024^{10} = 10^{30}$ (`Inf`!). If $\sigma_w = 0.01$, after 10 layers $\text{Var}(z) = 10^{-10}$ (`0.0`!).
     * **Xavier / Glorot Initialization:** For symmetric linear-near-zero activations (`tanh`), preserving forward variance ($\sigma_w^2 = \frac{1}{d_{\text{in}}}$) and backward gradient variance ($\sigma_w^2 = \frac{1}{d_{\text{out}}}$) gives $\sigma_w^2 = \frac{2}{d_{\text{in}} + d_{\text{out}}}$.
     * **Kaiming / He Initialization:** `ReLU(z) = max(0, z)` zeroes out exactly half of a symmetric distribution, cutting output variance in half! Therefore, to keep $\text{Var}(a_l) = 1$ across 100 layers, we **must** set $\sigma_w^2 = \frac{2}{d_{\text{in}}}$!
  3. **Why Activations Evolved:**
     * `Sigmoid` $\sigma(z) = \frac{1}{1+e^{-z}}$: Non-zero-centered outputs cause zig-zagging weight updates, and max derivative is $\sigma'(0) = 0.25$ ($0.25^{20} \approx 10^{-12}$ vanishing gradient!).
     * `Tanh`: Zero-centered, max derivative $1.0$, but still saturates to $0$ gradient when $|z| > 3$.
     * `ReLU`: Cheap hardware comparison, derivative $1$ for $z > 0$, **but** "Dead ReLU" problem when a large update pushes $z < 0$ permanently ($\text{grad} = 0$).
     * `GELU` & `SiLU` (`Swish`: $z \cdot \sigma(z)$): Smooth non-monotonic bend near zero prevents dead neurons and gives smoother optimization curvature.
     * **`SwiGLU`** ($\text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}}) W_{\text{down}}$): Multiplicative gating allows multiplicative interaction between features (Why Llama, Gemma, DeepSeek, and PaLM all replaced vanilla MLP with `SwiGLU`—and why intermediate width is scaled by $\frac{2}{3} \cdot 4d$ to keep parameter count and FLOPs matched!).
* **⚡ Practical Magic Lab (`labs/02_signal_propagation_diagnostics.py`):**
  * Build a 50-layer deep MLP in NumPy. Plot layer-by-layer forward activation histograms, backward gradient histograms, and gradient-to-data ratios (`|lr * grad| / |weight|`) under Naive $\mathcal{N}(0,1)$, Naive $\mathcal{N}(0, 0.01)$, Xavier, and Kaiming init—watching dead/exploding networks come cleanly to life!

---

### Module 1.3: Normalization Layers & Residual Gradient Highways
*How researchers scaled networks from 8 layers (AlexNet) to 152+ layers (ResNet) and 120+ layers (Transformers).*

* **Concepts Built From Scratch & Why Decisions Changed:**
  1. **Batch Normalization (`BatchNorm`):** Normalizing each feature across the mini-batch $(B, H, W)$ + learnable $\gamma, \beta$ + exponential moving average running stats for inference.
     * *Derive the full analytical backward pass of BatchNorm on paper!* Notice why $\frac{\partial L}{\partial x_i}$ has **three** coupled terms (direct path, path through $\mu$, path through $\sigma^2$).
     * **🔍 Why Did Transformers Abandon `BatchNorm`?** (1) Variable sequence lengths & padding tokens corrupt batch statistics; (2) Small per-GPU micro-batches in distributed training cause noisy batch stats; (3) Train vs Eval discrepancy (`running_mean`) creates subtle bugs in autoregressive generation and RL!
  2. **Layer Normalization (`LayerNorm`):** Normalizes across the *feature dimension* $D$ independently for every single token—zero dependence on batch size $B$ and identical behavior in train & eval!
  3. **Root Mean Square Normalization (`RMSNorm`):**
     $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2 + \epsilon}} \odot \gamma$$
     * **🔍 Why Did Modern LLMs Switch from `LayerNorm` to `RMSNorm`?** Empirically, re-centering mean $\mu$ adds zero accuracy benefit, only re-scaling invariance matters—yet computing *both* mean $\mu$ and variance $\sigma^2$ requires extra passes/reductions over memory! Since normalization is **100% memory-bandwidth bound** on GPUs (Phase 05), dropping $\mu$ cuts work and simplifies fused kernels with zero loss in quality!
  4. **Residual Skip Connections ($x_{l+1} = x_l + F_l(x_l)$) & Pre-LN vs Post-LN:**
     * Unroll $x_L = x_0 + \sum_{l=0}^{L-1} F_l(x_l)$. Differentiate with respect to $x_0$:
       $$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} \left( I + \frac{\partial}{\partial x_0} \sum_{l=0}^{L-1} F_l(x_l) \right)$$
     * That identity matrix **$I$** is an **unobstructed superhighway** carrying loss gradients straight from layer 100 back to layer 0 without multiplying through 100 weight matrices!
     * **🔍 Why Pre-LN Beat Post-LN:** Original 2017 Vaswani Transformer put `LayerNorm` *on* the residual trunk (`Post-LN`: `x = LN(x + F(x))`), attenuating the identity highway at init and requiring delicate warmup. Moving `RMSNorm` *inside* the branch (`Pre-LN`: `x = x + F(RMSNorm(x))`) keeps the trunk completely clean from embedding to final logits!
* **⚡ Practical Magic Lab (`labs/03_norms_and_residual_highways.py`):**
  * Implement `BatchNorm1d`, `LayerNorm`, and `RMSNorm` (forward + exact analytical backward) in NumPy, verify with `gradcheck`, and run a 40-layer network training comparison: Plain MLP vs Residual Post-LN vs Residual Pre-RMSNorm, plotting gradient norm at Layer 1 vs Layer 40!

---

### Module 1.4: Optimization Dynamics — From `SGD` to `AdamW` & Preconditioners
*Connecting your Phase 00 ravine conditioning insights to modern optimizer engineering.*

* **Concepts Built From Scratch:**
  1. **Mini-Batch `SGD`:** Gradient noise scale as implicit regularization vs ravine oscillation.
  2. **Heavy-Ball Momentum & Nesterov Accelerated Gradient:** Physics analogy—a heavy ball accumulates velocity along consistent downhill directions (valley floor) while opposing oscillations across steep ravine walls cancel out!
  3. **Adaptive Per-Parameter Scaling (`AdaGrad` $\to$ `RMSProp` $\to$ `Adam`):**
     * Track first moment $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ and second uncentered moment $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$.
     * *Prove on paper why bias correction $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ is mandatory!* Without it, since $m_0 = 0$ and $\beta_2 = 0.999$, at step 1 $v_1 = 0.001 g_1^2$, making step size explode or warp wildly!
     * Geometric intuition: $\frac{1}{\sqrt{\hat{v}_t} + \epsilon}$ acts as a cheap **diagonal approximation to the inverse Hessian/Fisher matrix**, rescaling ill-conditioned axes toward $\kappa \approx 1$!
  4. **🔍 Why `Adam` + $L_2$ Penalty Was Broken $\to$ `AdamW` (Decoupled Weight Decay):**
     * In standard `SGD`, adding $\frac{\lambda}{2}\|\theta\|^2$ to the loss gives gradient $g_t + \lambda \theta_t$, which decays every weight proportionally by $-\eta \lambda \theta_t$.
     * Inside `Adam`, if you add $\lambda \theta_t$ into $g_t$, it gets **divided by $\sqrt{v_t}$**! Weights with large historical gradients get *less* regularization than weights with tiny gradients! `AdamW` fixes this by applying $-\eta \lambda \theta_t$ directly to the weights *outside* the adaptive gradient fraction.
  5. **Learning Rate Schedules & Gradient Clipping:** Why linear **Warmup** stabilizes early `AdamW` variance estimates $v_t$, why **Cosine Annealing** settles into wide flat minima, why **Global $L_2$ Norm Gradient Clipping** prevents spike divergence without changing gradient direction (unlike value clipping!), and a peek at matrix-orthogonalizing optimizers (**Shampoo / Muon** using Newton-Schulz iterations!).
* **⚡ Practical Magic Lab (`labs/04_optimizers_in_the_ravine.py`):**
  * Animate `SGD`, `Momentum`, `RMSProp`, `Adam`, and `AdamW` racing simultaneously across Beale's function, Rosenbrock's valley, and a high-condition-number quadratic bowl, then compare `Adam + L2` vs `AdamW` generalization on a noisy polynomial/MLP fit.

---

### Module 1.5: Spatial & Sequential Inductive Biases (CNNs, `im2col`, RNNs & SSMs)
*How architectural structure bakes symmetry into weights—and how systems engineers secretly turn convolutions into matrix multiplications.*

* **Concepts Built From Scratch:**
  1. **Convolutions (`Conv1d` / `Conv2d`):** Why Dense MLPs fail on images (parameter explosion + zero translation equivariance). Kernel size, stride, padding, dilation, receptive fields, and depthwise-separable convolutions.
  2. **The Systems Secret (`im2col` & `col2im`):** A naive `Conv2d` in C++/Python is **7 nested `for` loops** (terrible cache & SIMD performance!). Learn how **`im2col`** rearranges sliding image patches into columns of a matrix so the entire `Conv2d` executes as **one single highly optimized `GEMM` ($W_{\text{flat}} X_{\text{col}}$)**! *(Notice the classic systems trade-off: trading extra RAM duplication for massive compute throughput—until implicit-GEMM CUDA kernels in Phase 05 solve both!)*
  3. **Recurrent Networks (`RNN` $\to$ `LSTM` / `GRU`):** Backpropagation Through Time (`BPTT`), proving via Jacobian products $\prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}}$ why recurrent weights explode/vanish and how additive cell-state gates ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$) create temporal residual highways!
  4. **The Hardware Flaw of RNNs $\to$ Linear State-Space Models (`S4` / `Mamba` Intuition):** Why non-linear step-by-step recurrence $h_t = \tanh(W h_{t-1} + U x_t)$ cannot parallelize across sequence length $T$ during training, whereas *linear* recurrence $h_t = A_t h_{t-1} + B_t x_t$ is associative and can be computed in $O(\log T)$ parallel time on GPUs via **Parallel Prefix Scan** (which you will write in CUDA in Phase 05!).
* **⚡ Practical Magic Lab (`labs/05_im2col_cnn_and_bptt.py`):**
  * Implement `Conv2d` twice in NumPy: (a) naive nested loops, and (b) `im2col` + `np.matmul` (+ `col2im` backward pass!). Benchmark the massive speedup even in pure NumPy and train a tiny LeNet on MNIST digits!

---

### Module 1.6: Attention & Modern Transformer Architecture Ins and Outs
*Deconstructing every single line of a modern LLM (`GPT` / `Llama` / `DeepSeek` style) from scratch.*

* **Concepts Built From Scratch & Every Design Decision Explained:**
  1. **Tokenization from Scratch (`Byte-Pair Encoding` / BPE):** Why character-level sequences are too long ($T$ explodes) and word-level vocabularies fail on rare/code tokens $\to$ subword BPE merge algorithm over UTF-8 bytes!
  2. **Scaled Dot-Product Self-Attention:**
     $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top + M}{\sqrt{d_k}}\right) V$$
     * *Intuition:* Differentiable content-addressable hash table (`Query` asks *"what am I looking for?"*, `Key` advertises *"what do I contain?"*, `Value` delivers *"here is my payload"*).
     * **🔍 Statistical Proof of $\frac{1}{\sqrt{d_k}}$:** If components of $q, k \in \mathbb{R}^{d_k}$ are independent with mean $0$ and variance $1$, then $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean $0$ and **variance $d_k$** (standard deviation $\sqrt{d_k}$)! For $d_k = 128$, dot products swing to $\pm 30$, pushing `Softmax` into `[0.0, 0.0, 1.0, 0.0]` where the Softmax Jacobian is **identically zero**! Dividing by $\sqrt{d_k}$ restores unit variance $\text{Var} = 1$!
  3. **Causal Masking ($M_{i,j} = -\infty$ for $j > i$):** Why autoregressive training processes all $T$ tokens in parallel in 1 forward pass using a triangular mask, whereas generation must step 1 token at a time!
  4. **Positional Information (`Learned` $\to$ `Sinusoidal` $\to$ `RoPE`):**
     * Attention without position is **permutation equivariant** (a bag of words!).
     * **🔍 Why `RoPE` (Rotary Position Embedding) Won:** Instead of adding position vectors to embeddings at layer 0, `RoPE` pairs dimensions $(2i, 2i+1)$ of $q_m$ and $k_n$ and rotates them in 2D by angles $m\theta_i$ and $n\theta_i$:
       $$\langle R_{m\theta} q, R_{n\theta} k \rangle = q^\top R_{(n-m)\theta} k$$
       Because the transpose of a rotation by $m\theta$ is a rotation by $-m\theta$, the dot product **automatically depends only on relative distance $(n - m)$**, preserves vector norms, and decays naturally with distance!
  5. **Multi-Head Attention (`MHA`) $\to$ Grouped-Query Attention (`GQA`) & `MLA`:**
     * Why multiple heads? Projecting into $H$ distinct subspaces lets Head 1 track syntax, Head 2 track previous token, Head 3 track induction patterns (`[A][B] ... [A] -> [B]`).
     * **🔍 Systems Foresight — Why `GQA` Replaced `MHA`:** During LLM inference, you cache past $K$ and $V$ tensors (**KV-Cache**). Sharing 1 Key/Value head across 4 or 8 Query heads (`GQA`) slashes KV-Cache VRAM and HBM memory bandwidth by **4x–8x** with almost zero perplexity loss!
* **⚡ Practical Magic Lab (`labs/06_transformer_and_bpe_numpy.py`):**
  * Write a 60-line UTF-8 BPE tokenizer from scratch.
  * Write analytical forward + backward passes for `RoPE`, `CausalMultiHeadAttention` (and `GQA`), `SwiGLU`, and `RMSNorm` in pure NumPy. Train a tiny Transformer on an algorithmic sequence task (e.g., reversing sequences or modular addition "grokking") and plot the **attention heatmaps** learning exact induction/copy heads!

---

### Module 1.7: Generative Paradigms & Deep Reinforcement Learning Foundations
*Rounding out the theoretical landscape: latent-variable statistics, score-based physics, and sequential decision-making for game/RL engines.*

* **Concepts Built From Scratch:**
  1. **Variational Autoencoders (`VAEs`):** Why intractable marginal likelihood $p_\theta(x) = \int p_\theta(x \mid z)p(z)dz$ cannot be computed directly; deriving the **Evidence Lower Bound (`ELBO`)** via Jensen's / KL decomposition:
     $$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z\mid x) \parallel p(z))}_{\text{ELBO}} + D_{\text{KL}}(q_\phi(z\mid x) \parallel p_\theta(z\mid x))$$
     * **The Reparameterization Trick ($z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$):** How a statistician moves stochasticity onto a leaf node so deterministic backprop flows cleanly through $\mu$ and $\sigma$!
  2. **Denoising Diffusion & Flow Matching Intuition:** Forward Ornstein-Uhlenbeck Gaussian corruption $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$, predicting the noise $\epsilon_\theta(x_t, t)$ ($\equiv$ estimating the **Stein Score** $\nabla_x \log p_t(x)$!), and straight-line ODE **Flow Matching**.
  3. **Deep Reinforcement Learning Foundations (Bridge to RL Systems!):**
     * Markov Decision Processes (`MDPs`: $\mathcal{S}, \mathcal{A}, P, R, \gamma$), discounted returns $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$, State-Value $V^\pi(s)$ & Action-Value $Q^\pi(s,a)$, and the **Bellman Optimality Equation**.
     * **Value-Based (`Tabular Q-Learning` $\to$ `DQN`):** Why naive neural Q-learning diverges (moving target + correlated frames) $\to$ Experience Replay Buffer + Target Network!
     * **Policy Gradients (`REINFORCE` $\to$ `Actor-Critic` $\to$ `PPO`):**
       * Derive the **Log-Derivative Trick** (Score Function Estimator) on paper—allowing gradients through *non-differentiable* C++ game/physics simulators!
         $$\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t\right]$$
       * Prove statistically why subtracting a state-dependent **Baseline $V(s_t)$** (`Advantage` $A_t = Q_t - V(s_t)$) keeps the estimator **100% unbiased** ($\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi(a\mid s) \cdot b(s)] = 0$!) while drastically slashing Monte Carlo variance!
       * **Why `PPO` (`Proximal Policy Optimization`) Exists:** If you take too big a step on policy parameters $\theta$, the state visitation distribution shifts violently and performance collapses irreversibly; clipping the importance-sampling ratio $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$ to $[1-\epsilon, 1+\epsilon]$ enforces a safe statistical trust region!
* **⚡ Practical Magic Lab (`labs/07_diffusion_spirals_and_ppo_lander.py`):**
  * **Part 1:** Train a tiny 3-layer Flow Matching / Diffusion score network in NumPy on a 2D Swiss Roll distribution and generate an animation watching pure Gaussian static condense into a crisp Swiss Roll!
  * **Part 2:** Build a custom 50-line 2D continuous-physics **Rocket Hover / CartPole** simulator in pure NumPy and train a pure-NumPy `REINFORCE` / `Actor-Critic` agent from scratch until it balances effortlessly!

---

## 📝 Required `notes/` Artifacts for Phase 01

1. `01_initialization_and_attention_variance_proofs.md` — Side-by-side statistical proofs of Xavier, Kaiming, and $\frac{1}{\sqrt{d_k}}$ scaled dot-product variance.
2. `02_norm_and_residual_backprop.md` — Hand derivation of `BatchNorm`, `LayerNorm`, and `RMSNorm` backward Jacobians + Pre-LN residual gradient highway proof.
3. `03_adamw_and_optimization_geometry.md` — Bias-correction derivation + proof of why $L_2$ weight penalty inside adaptive moments differs from decoupled `AdamW`.
4. `04_elbo_and_policy_gradient_theorems.md` — Complete paper derivation of the VAE ELBO and the unbiased zero-expectation baseline proof in Policy Gradients.

---

## 🏆 Phase 01 Boss Fight: `NumPyDL` — Zero-Dependency Deep Learning Micro-Framework

**Location:** `01_Deep_Learning_From_Scratch/labs/boss_fight_numpy_dl/`

**Rules of Engagement:**
* **Allowed core import:** `numpy` ONLY (`matplotlib` allowed for plots). Zero `torch`, zero `jax`, zero `autograd`.
* Every `Module` subclass implements explicit `.forward(x)` and `.backward(grad_out)` methods!

**Deliverables:**
1. **Modular Layer & Optimizer Library (`numpy_dl/`):**
   * Layers: `Linear` (with Kaiming/Xavier init), `Embedding`, `RMSNorm`, `SwiGLU` / `GELU`, `RoPE`, `CausalGQA`, `Conv2d` (`im2col`), and `SoftmaxCrossEntropy`.
   * Optimizer: `AdamW` with linear warmup + cosine learning rate schedule + global $L_2$ gradient norm clipping.
2. **Global Gradient Test Suite (`test_all_layers.py`):**
   * Automated centered finite-difference `float64` verification passing $< 10^{-7}$ max error across every single layer and parameter.
3. **Practical Showcase (Build Both Mini-Experiments):**
   * **Experiment A (Language):** Train a miniature `RoPE` + `RMSNorm` + `SwiGLU` character/BPE Transformer in pure NumPy on a small text corpus (e.g., Shakespeare / math expressions) and watch coherent text emerge!
   * **Experiment B (Game/RL):** Train an Actor-Critic policy using your `numpy_dl` engine inside a custom 2D vector physics arena and render an episode replay!

*Once you have built and trained a modern `RoPE`+`SwiGLU`+`RMSNorm` Transformer and an RL agent using nothing but NumPy and your own analytical calculus, no Deep Learning model on Earth will ever look like a "black box" to you again. Now—and only now—are you ready to dissect PyTorch and JAX from a position of total authority.*
