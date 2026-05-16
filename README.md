# 🧠 AI Systems Research Portfolio: From Statistics to Silicon

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-000000?style=for-the-badge&logo=Google&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

> **Mission Statement:** Modern AI is bottlenecked by hardware, not just math. This repository chronicles my transition from theoretical Statistics (BSc, 3.9 CGPA) to high-performance Machine Learning Systems. The goal is to build the capability to invent novel mathematical architectures and write the custom C++/CUDA engines required to execute them at scale.

## 🗺️ Repository Architecture

This workspace is strictly divided into five core phases, moving from high-level abstractions down to raw hardware execution. 

| Phase | Directory | Focus Area | Objective |
| :---: | :--- | :--- | :--- |
| **01** | `📁 01_PyTorch_and_JAX` | **Math Frontend** | Deconstructing the "Black Box." Implementing custom Autograd, vectorization (`vmap`), and JIT compilation from scratch. |
| **02** | `📁 02_Cpp_Engine` | **Systems Engine** | Mastering memory management. Pointers, references, heap allocation, and contiguous memory structures. |
| **03** | `📁 03_DL_Systems_Extensions` | **The Bridge** | Connecting Math to Hardware. Writing custom C++ tensor operations and binding them to Python via `Pybind11`. |
| **04** | `📁 04_CUDA_Accelerators` | **Parallelism** | Writing custom CUDA kernels. Matrix memory coalescing, shared memory tiling, and thread-grid architecture. |
| **05** | `📁 05_Capstone_Project` | **Portfolio Target** | *TBA:* A hyper-optimized, custom C++/CUDA implementation of a complex statistical algorithm (e.g., MCMC or Custom Optimizer). |
| **🔬** | `📁 MATLAB_Scratchpad` | **Verification** | Ground-truth verification using MATLAB's Stats and Symbolic Math Toolboxes to ensure C++ mathematical accuracy. |

---

## 🚀 Current Progress Tracker

- [ ] **Phase 1:** PyTorch internals, DAGs, and JAX Functional Paradigm.
- [ ] **Phase 2:** C++ memory architecture, pointers, and the STL.
- [ ] **Phase 3:** PyTorch C++ Extensions and Deep Learning Systems (CMU 10-414).
- [ ] **Phase 4:** CUDA massively parallel programming (PMPP 4th Ed).
- [ ] **Phase 5:** Capstone Statistical Algorithm Implementation.

---

## 🛠️ Environment & Setup

This repository uses a unified environment to ensure seamless bridging between the Python frontend and the C++ compiler.

**Core Tech Stack:**
*   **Languages:** Python (3.10+), Modern C++ (C++17/20), CUDA C++
*   **Frameworks:** PyTorch, JAX
*   **Build Tools:** CMake, Pybind11
*   **IDEs:** JetBrains DataSpell (Python) & CLion (C++)

**To reproduce this environment locally:**
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AI_Systems_Journey.git
cd AI_Systems_Journey

# 2. Recreate the environment
conda create --name ai_systems python=3.10
conda activate ai_systems

# 3. Install dependencies
pip install -r requirements.txt