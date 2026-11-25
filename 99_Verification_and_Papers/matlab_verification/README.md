# 🧮 Independent Mathematical Ground-Truth Verification

> **The Systems Researcher's First Commandment:** *A kernel that runs 100x faster while computing the wrong gradient is just a fast random number generator.*

Whenever you implement a custom backward pass, probability distribution, MCMC sampler, covariance kernel, or low-precision CUDA operator across **Phases 01–07**, verify numerical correctness against an independent oracle **before** optimizing performance:

## Three Tiers of Ground-Truth Oracles (Centered Around MATLAB + VS Code)

1. **MATLAB Symbolic Math & Matrix Oracle (Primary):**
   * Use **MATLAB's Symbolic Math Toolbox** (`syms`, `jacobian`, `hessian`, `simplify`) to verify every hand-derived matrix/tensor derivative before typing Python or C++ code.
   * Use **MATLAB's Statistics and Machine Learning Toolbox** to generate exact `float64` reference outputs for custom distributions, GP covariance kernels, and MCMC diagnostics.
   * **Seamless VS Code $\leftrightarrow$ MATLAB Bridge:** Save deterministic test inputs/outputs using `scipy.io.savemat("fixture.mat", {"x": x, "grad_ref": grad})` in Python and `load("fixture.mat")` in MATLAB (or vice-versa!) so your `pytest` suite in VS Code and Colab checks C++/CUDA kernels directly against MATLAB ground truth:
     * `< 1e-7` relative error for `float64`
     * `< 1e-5` relative error for `float32` (standard ops) / `< 1e-4` for Tensor Core `TF32`
     * `< 1e-2` to `1e-3` relative error for `FP16` / `BF16`
2. **Python Secondary Oracle (`SymPy` / `mpmath`):**
   * Handy inside **Google Colab** notebooks when running automated cloud CI checks without a local MATLAB session attached.
3. **Centered Finite-Difference Gradient Check (`float64`):**
   $$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \epsilon e_i) - L(\theta - \epsilon e_i)}{2\epsilon}, \qquad \epsilon = 10^{-6}$$
   * Always run gradient checks in **double precision (`float64`)**! In `float32`, machine epsilon is $\sim 1.19 \times 10^{-7}$, so subtracting $L(\theta + 10^{-6}) - L(\theta - 10^{-6})$ loses half its significant bits to catastrophic cancellation (Phase 00 Module 0.1!).
