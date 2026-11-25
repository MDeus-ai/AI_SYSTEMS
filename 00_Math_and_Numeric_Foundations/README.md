# 📐 Phase 00: First-Principles Math, Statistics, Floats & Scalar Autograd

> **Prerequisites:** **None.** Zero assumed math prerequisites—even for topics you saw in your Statistics BSc. We rebuild every concept from geometric, computational, and bit-level intuition so there are zero "trust-me" formulas anywhere in your foundation.  
> **Estimated Pace:** 3–5 Months (Self-paced; governed by passing Boss Fight 00).  
> **Tools Used in Phase 00:** **VS Code** (pure Python / NumPy / Matplotlib) + **MATLAB** (Symbolic Math & Statistics Toolboxes for checking analytical Jacobians/Hessians, SVD factorizations, and rotatable 3D loss/ravine surfaces) + **Graphviz** (rendering `MicroEngine` DAGs).  
> **Big Idea (1 Sentence):** *Before a neural network is "artificial intelligence," it is approximate base-2 floating-point numbers undergoing geometric linear transformations, steered down high-dimensional probability loss surfaces via reverse-mode chain rule.*

---

## 🎮 The Game Engine & Systems Connection
Why start with IEEE-754 bits, geometric vectors, and scalar calculus graphs?
* In a game engine, misunderstanding floating-point precision causes **"jitter" when a player walks far from the world origin `(0,0,0)`**; in Deep Learning, the exact same IEEE-754 mantissa/exponent trade-off causes **gradient underflow in `FP16`**, **overflow in naive `Softmax`**, and explains why Google invented **`BF16` (Brain Float 16)**!
* In a 3D engine, matrices rotate, scale, and project meshes into camera space; in Deep Learning, weight matrices rotate, stretch, and project data manifolds so curved classes become linearly separable.

> **🛠️ Phase 00 Tooling Setup:** Write all Python labs and `MicroEngine` in **VS Code** (with AI autocomplete turned off), and keep **MATLAB** open side-by-side as your interactive mathematical laboratory—use MATLAB's matrix environment (`svd`, `eig`, `cond`), Symbolic Math Toolbox (`jacobian`, `hessian`), and Statistics Toolbox to sanity-check every hand derivation!

---

## 📚 Primary Learning Resources for Phase 00

* **Visual Intuition (Watch First, Derive Second):**
  * Grant Sanderson (3Blue1Brown) — [Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) & [Essence of Calculus](https://www.3blue1brown.com/topics/calculus)
  * Brown University — [Seeing Theory: A Visual Introduction to Probability and Statistics](https://seeing-theory.brown.edu/)
* **Core Textbooks & Notes (Free Official Editions):**
  * Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong — [*Mathematics for Machine Learning* (Cambridge University Press)](https://mml-book.github.io/) *(Chapters 2–7)*
  * Gilbert Strang — [MIT 18.06 Linear Algebra Lectures](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) & [MIT 18.065 Matrix Methods in Data Analysis and ML](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)
  * Terence Parr & Jeremy Howard — [*The Matrix Calculus You Need For Deep Learning*](https://explained.ai/matrix-calculus/)
  * Chris Olah — [*Calculus on Computational Graphs: Backpropagation*](https://colah.github.io/posts/2015-08-Backprop/) & [*Visual Information Theory*](https://colah.github.io/posts/2015-09-Visual-Information/)
  * Joe Blitzstein — [Harvard Stat 110: Probability](https://projects.iq.harvard.edu/stat110/home) *(for deep statistical intuition refreshers)*
  * David Goldberg — *"What Every Computer Scientist Should Know About Floating-Point Arithmetic"* (read alongside [float.exposed](https://float.exposed/) interactive bit-flipper!)
* **From-Scratch Code Companion:**
  * Andrej Karpathy — [Neural Networks: Zero to Hero — Lecture 1: `micrograd`](https://karpathy.ai/zero-to-hero.html) *(watch ONLY up to understanding the problem, then close the video and build it yourself!)*

---

## 🧱 Chronological Modules

### Module 0.1: Numbers on Silicon — Binary & IEEE-754 Computer Arithmetic
*Real numbers ($\mathbb{R}$) are infinite and continuous; computer RAM is finite and discrete. Every mysterious `NaN` and quantization trick in Phase 06 begins right here.*

* **Concepts Built From Scratch:**
  1. Bits, bytes, unsigned integers, and two's complement signed integers (`INT8` range: `[-128, 127]`).
  2. Fixed-point vs Floating-point representation (scientific notation in base 2):
     $$\text{value} = (-1)^{\text{sign}} \times 2^{(\text{exponent} - \text{bias})} \times \left(1 + \frac{\text{mantissa}}{2^M}\right)$$
  3. Exact bit budgets across ML formats:
     * `FP32` (1 sign, 8 exponent, 23 mantissa)
     * `FP16` (1 sign, 5 exponent, 10 mantissa — max value `65,504`!)
     * `BF16` (1 sign, 8 exponent, 7 mantissa — **same dynamic range as `FP32`!**)
     * `FP8` (`E4M3` for forward weights/activations vs `E5M2` for wider-range gradients!)
  4. Machine epsilon ($\epsilon_{\text{mach}}$), non-associativity of float addition ($(a + b) + c \neq a + (b + c)$—crucial when parallel GPU threads sum numbers in different orders!), catastrophic cancellation, underflow, and overflow.
* **🔍 Why Was This Decision Made?**
  * *Why did AI labs abandon `FP16` for `BF16` during LLM pretraining?* Because `FP16` only has 5 exponent bits, so any intermediate activation or gradient norm exceeding `65,504` instantly overflows to `Inf` $\to$ `NaN`, requiring awkward dynamic "Loss Scaling." `BF16` kept `FP32`'s 8 exponent bits while sacrificing mantissa precision—and noisy stochastic gradients don't care about the 4th decimal digit, they care about dynamic range!
  * *Why do we ALWAYS subtract $\max(x)$ in `Softmax`?* Naive $\frac{e^{x_i}}{\sum_j e^{x_j}}$ overflows `FP32` at $x_i = 89$ (`e^89 = Inf`) and `FP16` at $x_i = 11.1$! Multiplying top and bottom by $e^{-c}$ where $c = \max_j x_j$ gives:
    $$\text{Softmax}(x)_i = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}$$
    Now the largest exponent is $e^0 = 1$, making overflow mathematically impossible! Derive the **Log-Sum-Exp (`LSE`)** trick the exact same way.
* **⚡ Practical Magic Lab (`labs/01_ieee754_and_numerical_stability.py`):**
  * Write a pure-Python bit-inspector using `struct.pack` / `np.view(np.uint32)` that prints the exact sign, exponent, and mantissa bits of any float.
  * Prove float addition non-associativity by summing `1.0` with $10^7$ copies of `1e-8` in forward vs reverse order; then implement **Kahan Compensated Summation** and watch the lost precision magically return!
  * Trigger `NaN` in naive `FP16` softmax on `[12.0, 10.0, 8.0]`, then prove your shifted `safe_softmax` & `logsumexp` stay rock-solid at `[1000.0, 998.0, 996.0]`.

---

### Module 0.2: Linear Algebra as Geometry & Data Transformations
*Forget memorizing determinant rules. Every matrix is a physical machine that warps space.*

* **Concepts Built From Scratch:**
  1. Scalars, vectors (both arrows in space and ordered tuples in RAM), norms ($L_1$, $L_2$, $L_\infty$), dot products ($a \cdot b = \|a\|\|b\|\cos\theta$ — why cosine similarity and attention dot products measure alignment!).
  2. Linear combinations, span, linear independence, and **Basis Vectors** (changing coordinate systems!).
  3. Matrix-vector multiplication $Ax$ two ways:
     * *Row picture:* $m$ dot-product hyperplanes measuring $x$.
     * *Column picture (Strang's Golden Key):* A linear combination of the columns of $A$ weighted by $x$!
  4. Matrix-matrix multiplication $AB$ four ways (elementwise dot products, column-space combinations, row-space combinations, and **sum of rank-1 outer products** $\sum_k a_{:,k} b_{k,:}$ — **this 4th view is the exact mathematical secret behind tiled GPU GEMM in Phase 05 and LoRA adapters!**).
  5. Rank, Null Space, Column Space, Orthogonal Matrices ($Q^\top Q = I$, norm-preserving rotations—why orthogonal weight initialization preserves gradient norms!), Projections, and Determinants as signed volume scaling.
  6. **Eigenvectors & Eigenvalues ($Av = \lambda v$):** Axes that only stretch (no rotation). Why repeated multiplication $A^T x$ in RNNs or optimization explodes if $|\lambda_{\max}| > 1$ and vanishes if $|\lambda_{\max}| < 1$! Symmetric Positive Semi-Definite (PSD) covariance/Hessian matrices ($x^\top \Sigma x \ge 0$).
  7. **Singular Value Decomposition ($A = U \Sigma V^\top$):** Every matrix in the universe—even rectangular—is simply: **(1) Rotate ($V^\top$) $\to$ (2) Stretch along axes ($\Sigma$) $\to$ (3) Rotate ($U$)**! Low-rank approximation (Eckart-Young theorem).
* **⚡ Practical Magic Lab (`labs/02_geometric_linalg_and_svd.py`):**
  * **2D Game Sprite / Grid Warper:** Plot a 2D unit grid and a simple polygonal spaceship in Matplotlib. Apply shear, 2D rotation ($\begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$), singular rank-collapse, and SVD step-by-step ($V^\top$, then $\Sigma$, then $U$) so you literally *see* $A = U\Sigma V^\top$ warp space.
  * **Low-Rank Image Compression:** Load a grayscale photo as matrix $A \in \mathbb{R}^{H \times W}$, compute SVD, reconstruct it using only the top $k \in \{1, 5, 20, 50\}$ singular outer products $\sum_{i=1}^k \sigma_i u_i v_i^\top$, and plot storage bytes vs visual fidelity.

---

### Module 0.3: Calculus, Multivariable Surfaces & Automatic Differentiation Modes
*How do you steer 70 billion parameters simultaneously when you can only afford one forward pass and one backward pass?*

* **Concepts Built From Scratch:**
  1. Single-variable derivatives from first principles: $f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$, and why **centered finite differences** $\frac{f(x+h)-f(x-h)}{2h}$ have $O(h^2)$ error vs $O(h)$ (prove this via Taylor series!).
  2. Partial derivatives, the **Gradient** $\nabla_x f \in \mathbb{R}^n$ (why is $\nabla f$ orthogonal to contour lines and pointing toward steepest ascent? Prove via directional derivative $\nabla_u f = \nabla f \cdot u$!).
  3. **The Jacobian Matrix** $J \in \mathbb{R}^{m \times n}$ for vector-to-vector functions $f: \mathbb{R}^n \to \mathbb{R}^m$ (local linear approximation $f(x + \Delta x) \approx f(x) + J \Delta x$).
  4. **The Hessian Matrix** $H \in \mathbb{R}^{n \times n}$ of second partials: local quadratic Taylor model $f(x + \Delta x) \approx f(x) + \nabla f^\top \Delta x + \frac{1}{2}\Delta x^\top H \Delta x$, positive/negative eigenvalues, minima, maxima, and **high-dimensional saddle points**.
  5. Multivariable Chain Rule on Directed Acyclic Graphs (DAGs): summing across fan-out paths!
  6. **The Three Ways to Differentiate on a Computer:**
     * *Numerical Differentiation* (Finite differences — $O(n)$ forward evaluations! With $n = 10^9$ weights, 1 gradient step would take 30 years!).
     * *Symbolic Differentiation* (SymPy expression swell).
     * *Automatic Differentiation (AD):*
       * **Forward-Mode AD (Jacobian-Vector Product, `JVP`):** Propagates tangents $\dot{v} = J u$ left-to-right. $O(1)$ sweep when $f: \mathbb{R}^1 \to \mathbb{R}^m$ (e.g., parametric curves).
       * **Reverse-Mode AD (Vector-Jacobian Product, `VJP` / Backpropagation):** Propagates adjoints $\bar{u} = v^\top J$ right-to-left. **$O(1)$ sweep when $f: \mathbb{R}^n \to \mathbb{R}^1$ (a scalar Loss function!)**—at the cost of storing forward activations in RAM!
* **🔍 Why Was This Decision Made?**
  * *Why does training an LLM take ~4x–8x more GPU memory than running inference on the exact same LLM?* Because Reverse-Mode AD (`VJP`) computes $\frac{\partial (x \cdot w)}{\partial w} = x^\top \bar{y}$: to compute the gradient during the backward pass, hardware **must keep $x$ (every layer's intermediate activation!) alive in memory** from the forward pass until the backward pass reaches that node! That single calculus rule explains the entire memory wall of Deep Learning.
* **⚡ Practical Magic Lab (`labs/03_jvp_vs_vjp_and_hessians.py`):**
  * Implement Dual Numbers (`class Dual(val, dot)`) in 30 lines of Python to watch exact Forward-Mode AD work with operator overloading (`sin`, `exp`, `*`, `+`) with zero finite-difference approximation error!
  * Plot a 3D saddle surface ($z = x^2 - y^2$) and a Rosenbrock banana valley, overlay the analytical gradient vector field, and animate a particle rolling down via gradient steps.

---

### Module 0.4: Probability, Statistical Inference & Information Theory from First Principles
*Bridging your Statistics degree directly into the loss functions, regularizers, and generative models of Deep Learning.*

* **Concepts Built From Scratch:**
  1. Probability spaces, conditional probability, Bayes' Theorem, independence vs conditional independence.
  2. Continuous & discrete distributions essential to ML: Bernoulli, Categorical, Gaussian ($\mathcal{N}(\mu, \Sigma)$ — why diagonal vs full covariance matters in memory!), Laplace, Beta/Dirichlet, Gumbel (the **Gumbel-Max trick** for sampling categorical tokens!).
  3. Expectation, Variance ($\text{Var}(aX + b) = a^2\text{Var}(X)$, variance of sums of independent variables—you will use this in Phase 01 to derive Xavier/Kaiming initialization and $\frac{1}{\sqrt{d_k}}$ attention scaling!), Covariance matrices, and Monte Carlo Estimator variance ($\frac{\sigma^2}{N}$).
  4. **Statistical Estimation $\leftrightarrow$ Loss Functions:**
     * **Maximum Likelihood Estimation (MLE):** Prove on paper that maximizing Gaussian log-likelihood $\log \mathcal{N}(y \mid f_\theta(x), \sigma^2)$ is **identical** to minimizing **Mean Squared Error (MSE)**!
     * Prove that Laplace noise MLE gives **$L_1$ / MAE Loss** (and combine them to get **Huber Loss**!).
     * Prove that Bernoulli/Categorical log-likelihood gives **Binary / Categorical Cross-Entropy**!
     * **Maximum A Posteriori (MAP):** Prove that placing a zero-mean Gaussian prior $\theta \sim \mathcal{N}(0, \tau^2 I)$ yields **$L_2$ Regularization**, and a Laplace prior yields **$L_1$ Lasso sparsity**!
  5. **Information Theory (Claude Shannon's Lens):**
     * Self-information ("Surprise"): $I(x) = -\log p(x)$ (derived from the single axiom that independent events have additive surprise: $I(x, y) = I(x) + I(y)$!).
     * **Shannon Entropy:** $H(P) = \mathbb{E}_{x \sim P}[-\log P(x)]$.
     * **Cross-Entropy:** $H(P, Q) = \mathbb{E}_{x \sim P}[-\log Q(x)]$ — average bits needed to encode truth $P$ using model $Q$'s code!
     * **Kullback-Leibler (KL) Divergence:** $D_{\text{KL}}(P \parallel Q) = H(P, Q) - H(P) \ge 0$ (prove non-negativity via Jensen's Inequality!). Notice **asymmetry**: Mode-covering Forward KL $D_{\text{KL}}(P \parallel Q)$ vs Mode-seeking Reverse KL $D_{\text{KL}}(Q \parallel P)$ (used in RL/PPO and Variational Inference!).
* **⚡ Practical Magic Lab (`labs/04_mle_kl_and_monte_carlo.py`):**
  * Fit a toy neural curve to noisy heteroscedastic data where the network predicts *both* $\mu(x)$ and $\log \sigma^2(x)$ via exact Gaussian Negative Log-Likelihood, plotting live 95% epistemic/aleatoric uncertainty bands!
  * Plot Forward KL vs Reverse KL fitting a unimodal Gaussian $Q$ to a bimodal mixture $P$: *watch* Forward KL stretch across the empty middle while Reverse KL snaps cleanly onto a single peak!

---

### Module 0.5: Conditioning, Geometry of Optimization & Constrained Problems
*Why does pure gradient descent take 10,000 steps in a narrow valley when Newton's method takes 1 step—and why can't we use Newton's method on LLMs?*

* **Concepts Built From Scratch:**
  1. Convex sets & convex functions (Jensen's inequality), local minima vs saddle points in $d \gg 1000$ dimensions.
  2. Quadratic bowl analysis $f(x) = \frac{1}{2} x^\top Q x - b^\top x$: exact step-size stability bound $0 < \eta < \frac{2}{\lambda_{\max}(Q)}$!
  3. **The Condition Number ($\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$):** Why elongated elliptical valleys force tiny learning rates along steep walls while crawling agonizingly slowly along the flat valley floor! *(This single insight motivates Feature Normalization, BatchNorm/LayerNorm, Momentum, Adam, and Preconditioners in Phase 01!)*
  4. Newton's Method ($x_{t+1} = x_t - H^{-1}\nabla f$): $O(n^2)$ memory to store $H$ and $O(n^3)$ FLOPs to invert $H$—for $n = 10^9$ weights, $H$ would require $4 \times 10^{18}$ bytes (4 Exabytes!) of RAM!
  5. Constrained optimization & **Lagrange Multipliers** ($\nabla f = \lambda \nabla g$ — used directly to prove maximum-entropy distributions and PPO trust-region constraints).
* **⚡ Practical Magic Lab (`labs/05_condition_numbers_and_ravines.py`):**
  * Build an interactive contour visual comparing Gradient Descent on an isotropic bowl ($\kappa = 1$) vs an ill-conditioned ravine ($\kappa = 50$), testing $\eta$ just below and just above $\frac{2}{\lambda_{\max}}$ to watch exact mathematical divergence happen right on cue!

---

## 📝 Required `notes/` Artifacts for Phase 00

Create these four handwritten/Markdown derivations inside `00_Math_and_Numeric_Foundations/notes/`:
1. `01_float_anatomy.md` — Bit layout diagram comparing `FP32`, `FP16`, and `BF16` + full algebraic derivation of Safe Softmax and `LogSumExp`.
2. `02_svd_and_four_views_of_matmul.md` — Visual sketch + algebraic write-up of the 4 ways to view $C = AB$ (especially outer-product sum) and geometric meaning of $A = U\Sigma V^\top$.
3. `03_forward_vs_reverse_ad.md` — Complexity table and computational graph trace showing why `JVP` costs $O(n)$ passes and `VJP` costs $O(1)$ pass + activation storage for scalar losses.
4. `04_loss_functions_from_likelihood.md` — Clean statistical proofs deriving MSE, MAE, Cross-Entropy, and $L_2$ weight decay from MLE/MAP and KL Divergence.

---

## 🏆 Phase 00 Boss Fight: `MicroEngine` (From Pure Python `math` to Live Learning)

**Location:** `00_Math_and_Numeric_Foundations/labs/boss_fight_microengine/`

**Rules of Engagement:**
* **Allowed imports for the engine:** `math`, `random` ONLY (no `torch`, no `jax`, not even `numpy` inside the core scalar `Value` engine; `matplotlib` allowed only for visualization, and `numpy` allowed only if you build the bonus vector extension).
* **No AI code generation.**

**Deliverables:**
1. **The Autograd Core (`engine.py`):**
   * Build `class Value` wrapping a Python `float`, tracking `_prev` parents, `_op` label, and `grad`.
   * Implement `+`, `-`, `*`, `/`, `**`, `exp()`, `log()`, `tanh()`, and `relu()` with exact local analytical backward closures.
   * Handle the **variable reuse bug** (`a + a` or `a * a`) properly by *accumulating* (`+=`) gradients rather than overwriting (`=`)!
   * Implement topological sort via DFS and `.backward()`.
2. **Automated Numerical Gradient Verifier (`test_gradcheck.py`):**
   * Write a centered finite-difference tester $\frac{f(x+h) - f(x-h)}{2h}$ ($h = 10^{-6}$) that builds random composite expressions and asserts relative error $< 10^{-6}$ for every leaf node.
3. **DAG Computation Graph Renderer (`viz.py`):**
   * Output an SVG/ASCII visualization showing forward `.data` and backward `.grad` inside every node.
4. **Seeing Practical Magic (`train_moons_spiral.py`):**
   * Build `Neuron`, `Layer`, and `MLP` classes atop `Value`.
   * Generate a non-linearly separable **2D Two-Moons or Two-Spirals** dataset.
   * Train with Safe Cross-Entropy Loss + $L_2$ MAP Regularization.
   * Save a GIF / multi-panel figure watching the non-linear decision boundary bend and wrap cleanly around the spirals over epochs alongside the loss curve!

*Once your `test_gradcheck.py` passes with $< 10^{-6}$ error and your spiral decision boundary converges smoothly, you have conquered Phase 00 and earned entry to Phase 01.*
