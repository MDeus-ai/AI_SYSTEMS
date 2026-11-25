# 🔧 Phase 03: Modern C++20, Hardware Architecture & Game-Engine Data-Oriented Design

> **Prerequisites:** [Phases 00–02](../README.md) completed. Prior C++ syntax familiarity helps, **but zero advanced C++ or computer architecture background is assumed**—we build every memory, OS, cache, and CPU instruction concept from physical silicon up.  
> **Estimated Pace:** 4–6 Months.  
> **Tools Used in Phase 03:** **Visual Studio 2022** (`MSVC` C++20, native [CMakePresets.json](./CMakePresets.json) integration, Memory Hex Viewer, `Alt+8` Live Disassembly, AddressSanitizer & `Alt+F2` CPU Profiler) + **[Compiler Explorer (`godbolt.org`)](https://godbolt.org/)** + optional **Raylib / Tracy Profiler**.  
> **Big Idea (1 Sentence):** *Python is a polite illusion; on real silicon, moving bytes from DRAM through L3/L2/L1 caches into SIMD registers costs ~100x more time and energy than doing the math—so mastering C++20 memory layout, custom allocators, cache tiling, and AVX2 vectorization lets you build engines that run 50x–200x faster.*

---

## 🎮 Where Game Engine Dev & ML Systems Become One
This will likely be one of the **most fun phases of your entire life**. Why? Because high-performance C++ game engines (`Unreal`, `id Tech`, `Unity DOTS`, Casey Muratori's *Handmade Hero*) and high-performance ML runtimes (`PyTorch ATen`, `llama.cpp`, `MuJoCo`, `ONNX Runtime`) use the **exact same tricks**:
* **Data-Oriented Design (`SoA` over `AoS`):** Grouping hot numerical arrays contiguously so every 64-byte CPU cache line fetch feeds 16 `float32`s straight into math pipelines with zero wasted pointer chasing.
* **Custom Arena Allocators:** Eliminating heap `malloc`/`new` fragmentation inside 60-FPS game frames *and* inside neural forward/backward passes.
* **SIMD Intrinsics (`AVX2` / `FMA`):** Computing 8 particle physics updates—or 8 neuron dot products—in a single CPU clock cycle!

---

## 📚 Primary Learning Resources for Phase 03

* **Modern C++ (Clean Systems Subset — Skip Bloated OOP Hierarchies!):**
  * Alex — [LearnCpp.com](https://www.learncpp.com/) *(Focus on Pointers/References, `const`, Dynamic Allocation, Move Semantics, Smart Pointers, `std::vector` / `std::span`, Templates)*
  * Bjarne Stroustrup — *A Tour of C++ (3rd Edition)* *(Concise 250-page modern C++20 overview)*
  * [cppreference.com](https://en.cppreference.com/) *(Your daily old-school manual—bookmark it!)*
  * [Modern CMake Introduction](https://cliutils.gitlab.io/modern-cmake/)
* **Computer Architecture & High-Performance CPU Engineering:**
  * Randal E. Bryant & David R. O'Hallaron — *Computer Systems: A Programmer's Perspective (`CS:APP`, 3rd Ed / CMU 15-213)* *(Chapters 1, 3, 5, 6, 9 — **Chapter 5 "Optimizing Program Performance" and Chapter 6 "The Memory Hierarchy" are mandatory life-changing chapters!**)*
  * Sergey Slotin — [*Algorithmica: High-Performance Computing*](https://en.algorithmica.org/hpc/) *(Free online book covering Pipelining, SIMD AVX2, CPU Caches, and Fast Matrix Multiplication from scratch!)*
  * Ulrich Drepper — *"What Every Programmer Should Know About Memory"* *(Sections 1–3 & 6)*
* **Game Engine & Data-Oriented Design Gems:**
  * Mike Acton (Insomniac Games Engine Director) — CppCon 2014 Talk: *"Data-Oriented Design and C++"*
  * Casey Muratori — [*Performance-Aware Programming* (`computerenhance.com`)](https://www.computerenhance.com/)
  * Jason Gregory — *Game Engine Architecture (3rd Edition)* *(Chapter 5: Engine Support Systems / Memory Allocators, Chapter 6: SIMD Math)*
* **Essential Interactive Tool:**
  * Matt Godbolt — [Compiler Explorer (`godbolt.org`)](https://godbolt.org/) *(Paste C++ on the left, read exact `x86-64` assembly instructions like `vfmadd231ps` on the right!)*

---

## 🧱 Chronological Modules

### Module 3.1: The Compilation Pipeline, Visual Studio + CMake & Memory Sanitizers
*C++ does not "just run." Knowing how source text becomes `MSVC` machine opcodes saves hundreds of hours when building PyTorch/CUDA extensions in Phase 04.*

* **Concepts Built From Scratch:**
  1. **The 4 Stages of Compilation:**
     * *Preprocessing* (`#include`, `#define` text expansion) $\to$ *Compilation* (`.cpp` $\to$ `.asm` assembly, optimization flags Debug `/Od` vs Release `/O2 /arch:AVX2` in MSVC or `-O0` vs `-O3 -march=native` in Clang/GCC) $\to$ *Assembly* (`.obj`/`.o` machine code with unresolved symbols) $\to$ *Linking* (stitching `.obj` files and `.dll`/`.lib` libraries—understanding why `LNK2019: unresolved external symbol` is a **linker** error, not a syntax error!).
  2. **Visual Studio Native `CMake` Workflow:** Open `03_Cpp_and_Hardware_Architecture/` directly in **Visual Studio 2022** via *File $\to$ Open $\to$ Folder...* (or *CMake...*)—Visual Studio natively parses [CMakeLists.txt](./CMakeLists.txt) with zero `.sln` boilerplate! **Crucial Rule:** Always switch the Visual Studio toolbar configuration dropdown from **`x64-Debug` to `x64-Release`** before running any benchmark (Debug disables inlining and adds iterator bounds checks that make `std::vector` 20x slower!).
  3. **Visual Studio Bare-Metal Debugging Superpowers:**
     * **Memory Window** (*Debug $\to$ Windows $\to$ Memory $\to$ Memory 1*): Paste `&heap_tensor[0]` at a breakpoint to inspect raw little-endian IEEE-754 bytes and contiguous cache lines live in RAM!
     * **Disassembly Window** (`Alt+8` while paused at a breakpoint): Inspect the exact `x86-64` / `AVX2` instructions (`vfmadd231ps`) generated by `MSVC` right inline with your C++ source—and compare compilers anytime on [Compiler Explorer (`godbolt.org`)](https://godbolt.org/)!
     * **AddressSanitizer (`ASan`):** Enable `-DENABLE_ASAN=ON` (`/fsanitize=address` in Visual Studio!) to catch out-of-bounds heap/stack bugs the instant they happen.
     * **Visual Studio Performance Profiler (`Alt+F2`):** Sample CPU hot paths and cache stalls (or drop in the single-header **Tracy Profiler** for game-engine-style frame timelines).
* **⚡ Practical Magic Lab (`src/01_godbolt_and_cmake.cpp`):**
  * Open this folder in **Visual Studio**, build & run `hello_silicon` in `x64-Release` vs `x64-Debug` to witness the Debug-vs-Release speed chasm firsthand. Pause at a breakpoint in `Debug`, open *Memory 1* and *Disassembly* (`Alt+8`), then paste a 5-line array sum into `godbolt.org` (`x64 msvc v19.latest` with `/O2 /arch:AVX2`) and watch scalar `addss` turn into 256-bit `vaddps` SIMD instructions!

---

### Module 3.2: Virtual Memory, The Stack, The Heap & Zero-Copy `mmap`
*Where do 70 Gigabytes of LLM weights physically live, and how do modern engines load them in 0.05 seconds?*

* **Concepts Built From Scratch:**
  1. **The Stack:** Fast, LIFO bump-pointer register (`rsp`), ~1–8 MB size limit. Local scalars (`float lr = 1e-3f;`) live here and vanish in 0 nanoseconds when scope exits. Allocating `float weights[10000][10000]` on the stack instantly triggers a **Stack Overflow (`SIGSEGV`)**!
  2. **The Heap:** Massive GiB pool managed by the allocator (`new` / `malloc` / `std::vector` buffer), accessed via stack pointers pointing into heap addresses.
  3. **OS Virtual Memory & Page Tables (4 KB Pages):** Every process sees a clean virtual address space translated to physical RAM chips by the CPU's Memory Management Unit (`MMU`) + Translation Lookaside Buffer (`TLB`).
  4. **🔍 Systems Secret — Memory-Mapped Files (`mmap` / Windows `CreateFileMapping`):**
     * Naive weight loading: `fread()` copies 14 GB from SSD $\to$ OS Kernel Page Cache $\to$ User Heap Tensor (2x RAM spike + seconds of CPU copying!).
     * **`mmap` (How HuggingFace `safetensors` and `llama.cpp` work!):** Maps the file on SSD **directly** into your virtual address space as a `const float*` pointer in <1 millisecond with **zero copies**, paging 4 KB chunks from NVMe SSD straight into RAM on demand!
* **⚡ Practical Magic Lab (`src/02_stack_heap_and_mmap.cpp`):**
  * Write a C++ binary tensor writer/loader comparing standard `std::ifstream::read` vs OS memory-mapped `mmap`/`MapViewOfFile` on a 1 GB float binary file—measuring cold-start latency and peak RSS memory!

---

### Module 3.3: Pointers, `std::span`, `const` Correctness, RAII & Move Semantics
*Wielding raw hardware addresses with modern C++20 safety—zero memory leaks, zero accidental 10-GB copies.*

* **Concepts Built From Scratch:**
  1. **Raw Pointers & Pointer Arithmetic:** `&x` (address-of), `*ptr` (dereference), `ptr + k` advancing `k * sizeof(T)` bytes, byte alignment (`alignof`, `alignas(64)`).
  2. **Pass-by-Value vs Pass-by-`const`-Reference & `std::span<const float>`:**
     * *The OOM Trap:* `float variance(std::vector<float> x)` duplicates 400 MB on the heap!
     * *Modern C++20 Idiom:* `float variance(std::span<const float> x)` passes just `{pointer, size}` (16 bytes in two CPU registers!) working seamlessly over `std::vector`, raw arrays, `mmap` buffers, and PyTorch `data_ptr()`!
  3. **RAII (Resource Acquisition Is Initialization) & Smart Pointers:** Why manual `new`/`delete` is banned in modern C++; `std::unique_ptr<T[]>` (zero-overhead exclusive ownership) vs `std::shared_ptr<T>` (atomic reference counting—exactly how PyTorch `c10::intrusive_ptr<StorageImpl>` manages shared tensor views!).
  4. **Move Semantics (`std::move` & Rvalue References `T&&`):** Stealing the internal `{data_ptr, size, capacity}` 24-byte header of a 10-GB tensor instead of copying $2.5 \times 10^9$ floats!
* **⚡ Practical Magic Lab (`src/03_cost_of_copying_variance.cpp`):**
  * *(Your upgraded classic benchmark!)* Allocate a flattened $10{,}000 \times 10{,}000$ (`400 MB`) matrix. Compute Welford/two-pass statistical **Variance** via Pass-by-Value (`calc_var_copy`), Pass-by-Const-Ref (`calc_var_ref`), and `std::move` ownership transfer—proving the exact millisecond & bandwidth cost of copying!

---

### Module 3.4: CPU Cache Hierarchy, Data-Oriented Design (`AoS` vs `SoA`) & Arena Allocators
*The single biggest mental shift in systems & game engine programming.*

* **Concepts Built From Scratch:**
  1. **The Silicon Latency Chasm (Memorize Order of Magnitude!):**
     * CPU Register: **~0.3 ns** (1 cycle)
     * L1 Data Cache (32–48 KB per core): **~1 ns** (~4 cycles)
     * L2 Cache (512 KB–2 MB per core): **~4 ns** (~14 cycles)
     * L3 Shared Cache (16–64 MB): **~12–20 ns** (~50 cycles)
     * Main System DRAM: **~80–100 ns** (~300+ cycles — **an eternity while your 5 GHz ALU sits completely idle waiting for bytes!**)
  2. **64-Byte Cache Lines & Spatial Locality:** When you read `float x = ptr[0]` (4 bytes), hardware fetches the **entire 64-byte cache line** (`ptr[0]` through `ptr[15]`)!
     * Row-major traversal (`row * width + col` with `col` inner loop): 15 out of 16 reads are instant **L1 Cache Hits**!
     * Column-major traversal of a row-major matrix (`row` inner loop): jumps by `width * 4` bytes every step, trashing every cache line (**10x–20x slower for the exact same number of FLOPs!**).
  3. **Game Engine Secret #1 — `AoS` vs `SoA`:**
     * *OOP Array of Structures (`AoS`):* `struct Entity { float x, y, vx, vy; char name[64]; int health; double inventory[16]; };` Updating positions `x += vx * dt` pulls `name`, `health`, and `inventory` into your precious L1 cache, wasting **92% of memory bandwidth**!
     * *Data-Oriented Structure of Arrays (`SoA` = Tensor Layout!):* Separate contiguous arrays `float* x, *y, *vx, *vy`—**100% cache-line utilization** and automatic SIMD readiness!
  4. **Game Engine Secret #2 — Custom Arena (Bump) Allocator:**
     * Build `class ArenaAllocator` in 50 lines of C++: `malloc` one 64-byte-aligned block upfront; `alloc(bytes)` simply aligns `offset` and bumps `offset += bytes` in **2 CPU instructions**, and `reset()` clears every temporary tensor/frame allocation at the end of a step in **$O(1)$ (`offset = 0`)**!
* **⚡ Practical Magic Lab (`src/04_cache_strides_and_arena.cpp`):**
  * Plot memory access throughput (GB/s) across buffer sizes from `4 KB` to `256 MB` with stride 1 vs random pointer-chasing (`std::mt19937` permutation)—**your plot will physically reveal the exact KB/MB sizes of your own laptop CPU's L1, L2, and L3 caches!**
  * Benchmark allocating 1,000,000 small temporary buffers with `new`/`delete` vs your custom `ArenaAllocator`.

---

### Module 3.5: The Roofline Model & Rigorous Microbenchmarking
*How frontier systems researchers tell in 10 seconds whether an algorithm needs better math or better memory layout.*

* **Concepts Built From Scratch:**
  1. **Hardware Ceiling #1 — Peak Compute ($\Pi$, GFLOP/s):** $\text{Cores} \times \text{Clock (GHz)} \times \text{SIMD Width} \times 2 \text{ (FMA)}$.
  2. **Hardware Ceiling #2 — Peak Memory Bandwidth ($\beta$, GB/s):** Measured via STREAM triad benchmark (`a[i] = b[i] + s * c[i]`).
  3. **Arithmetic Intensity ($\text{AI} = \frac{\text{Total FLOPs}}{\text{Total Bytes Moved}}$) & The Roofline Formula:**
     $$\text{Attainable Performance (GFLOP/s)} = \min\left(\Pi, \; \beta \times \text{AI}\right)$$
     * **Vector Add (`c = a + b`):** 1 FLOP per 12 bytes read/written $\implies \text{AI} = \frac{1}{12} \approx 0.083$ FLOP/Byte (**100% Memory-Bandwidth Bound!** Buying a faster ALU does nothing!).
     * **Matrix Multiply ($N \times N$):** $2N^3$ FLOPs over $3 \times 4N^2$ bytes $\implies \text{AI} = \frac{N}{6}$ FLOP/Byte! For $N = 1024$, $\text{AI} \approx 170$ FLOP/Byte (**Compute Bound!**—*if and only if* your caches reuse tiles instead of re-fetching from DRAM!).
  4. **Defeating Compiler Cheating in Benchmarks:** Using [include/bench_timer.hpp](./include/bench_timer.hpp) (`do_not_optimize` inline assembly barrier) so `-O3` doesn't delete your benchmarked loop when the output isn't printed!

---

### Module 3.6: Breaking the CPU Speed Limit — ILP, `AVX2` SIMD Intrinsics & Multithreading
*Turning one CPU core into an 8-lane vector beast, then scaling across all cores without cache-line ping-pong.*

* **Concepts Built From Scratch:**
  1. **Instruction-Level Parallelism (`ILP`) & Dependency Chains:** Why summing an array into *one* accumulator `sum += a[i]` stalls on a 4-cycle FP-add latency pipeline, whereas unrolling into **4 independent accumulators** (`sum0..sum3`) instantly runs **4x faster on a single thread**!
  2. **SIMD Intrinsics (`#include <immintrin.h>`):**
     * Loading 8 floats (`_mm256_loadu_ps`), broadcasting scalars (`_mm256_set1_ps`), and **Fused Multiply-Add (`_mm256_fmadd_ps(a, b, c)`)**—computing $a \cdot b + c$ on 8 floats simultaneously (16 FLOPs per instruction!).
  3. **Multithreading (`OpenMP` / `std::jthread`) & The False Sharing Trap:**
     * Parallelizing outer loops with `#pragma omp parallel for`.
     * **False Sharing:** Why two CPU cores writing to `results[thread_0]` and `results[thread_1]` right next to each other on the **same 64-byte cache line** cause catastrophic L1 cache invalidation storms (10x slowdown!), and how `alignas(64)` fixes it instantly!

---

## 📝 Required `notes/` Artifacts for Phase 03

1. `01_cpu_memory_and_cache_map.md` — Diagram of Virtual Memory (`mmap` + Stack + Heap) feeding DRAM $\to$ L3 $\to$ L2 $\to$ L1 (64B lines) $\to$ 256-bit `YMM` AVX2 Registers, annotated with your own measured cache cliff sizes!
2. `02_tensor_strides_and_soa_math.md` — Row-major vs Column-major 2D (`r * W + c`) & 3D (`d * H * W + r * W + c`) stride derivations + why `SoA` and contiguous tensors are identical.
3. `03_my_cpu_roofline_spec.md` — Exact calculation of your machine's theoretical Peak GFLOP/s ($\Pi$), measured STREAM GB/s ($\beta$), and Ridge Point ($\Pi / \beta$).

---

## 🏆 Phase 03 Two-Part Boss Fight: Game-Engine DOD + Silicon GEMM

**Location:** `03_Cpp_and_Hardware_Architecture/src/`

### Part A (The Game Engine Joyride): 100,000-Particle 2D Physics & Boids Engine
1. Write **Version 1 (Classic OOP `AoS`)**: `std::vector<std::unique_ptr<Particle>>` scattered on the heap with virtual `update(dt)` methods, gravity wells, and boundary collisions. Measure ms/frame (FPS).
2. Write **Version 2 (Data-Oriented `SoA` + `ArenaAllocator` + `AVX2` SIMD + `OpenMP`)**: Zero heap pointer chasing, 64-byte aligned contiguous `x, y, vx, vy` buffers, `_mm256_fmadd_ps` physics integrator.
3. Render a visual frame dump (PPM image or lightweight SDL2/Raylib window if desired) and log the **50x–150x FPS speedup**!

### Part B (The ML Engine Core): Climbing the CPU GEMM Optimization Ladder
Compute $C = A \times B$ for $1024 \times 1024$ `float32` matrices ($2.15 \times 10^9$ FLOPs) in 5 progressive C++ kernels, verifying all 5 against naive `float64` to $< 10^{-4}$:
1. **Kernel 1:** Naive `i-j-k` triple loop (~0.5–2 GFLOPS — cache-thrashing `B` column reads).
2. **Kernel 2:** Loop-reordered `i-k-j` (~8–15 GFLOPS — sequential spatial locality!).
3. **Kernel 3:** L1/L2 Cache-Blocked Tiling (`32x32` / `64x64` tiles).
4. **Kernel 4:** `AVX2` + `FMA` $6 \times 16$ Register Micro-Kernel (keeping `C` accumulators pinned in `YMM` registers!).
5. **Kernel 5:** `OpenMP` Multithreaded + Packed Cache-Tiled `AVX2` Kernel (**target: >80–200+ GFLOPS!**).
*Plot all 5 kernels as dots climbing straight up your **CPU Roofline Chart**!*
