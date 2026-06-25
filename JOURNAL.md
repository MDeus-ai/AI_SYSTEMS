# 📔 The Systems Log (Daily Journal)

> **The Rule:** Write in this journal at the end of every coding session. It doesn't matter if I coded for 4 hours or 15 minutes. Document the struggle, and document the victory. 

---

## 📝 Blank Entry Template
*(Copy and paste this block for each new day)*

### 📅 [YYYY-MM-DD] | Phase [X]: [Topic]
* **🎯 Goal Today:** [What did I sit down to achieve?]
* **✅ Accomplished:**[What actually got done?]
* **🐛 The Struggle (Bug/Math Wall):** [What confused me or broke?]
* **💡 The Fix / Insight:** [How did I overcome it? What was the "Aha!" moment?]

---

## 📖 Log Entries
### 📅 2026-06-15 | Phase 1: Tensor Memory Layout
* **🎯 Goal Today:** Understand memory layouts
* **✅ Accomplished:** Understood the concept of storages, that tensors are basically stored as a 1D array of numerical data.
* **🐛 The Struggle (Bug/Math Wall):** What offset actually does in practise
* **💡 The Fix / Insight:** Understood that an offset is the index in the storage corresponding to the first element in the tensor

### 📅 2026-06-16 | Phase 1: Tensor Memory Layout
* **🎯 Goal Today:** Deconstruct Offsets
* **✅ Accomplished:** Understood that Pytorch does not make a copy of a tensor when ever its derived from another tensor, instead it simply adjusts the metadata (shape, stride and offset) to describe this sub-tensor
* **🐛 The Struggle (Bug/Math Wall):** None
* **💡 The Fix / Insight:** Understood that Pytorch does not make a copy of a tensor when ever its derived from another tensor, instead it simply adjusts the metadata (shape, stride and offset) to describe this sub-tensor

### 📅 2026-06-17 | Phase 1: Tensor Memory Layout
* **🎯 Goal Today:** Understand Contiguity
* **✅ Accomplished:** Understood that pytorch uses the Row-Major Order to store and read data
* **🐛 The Struggle (Bug/Math Wall):** None
* **💡 The Fix / Insight:** Understood that a tensor is contiguous but its transpose is not because of the change in the rules (strides, size) on how pytorch access its contents. Since pytorch accesses data using Row-Major Order, it forces it to jump back and forth in the memory reducing performance

### 📅 2026-06-22 | Module 2: Math Operations & Early Measurement Culture
* **🎯 Goal Today:** Understand how to accurately measure and benchmark tensor execution speed using Python's `timeit` library.
* **✅ Accomplished:** Learned the mechanics of `timeit` (runs nested loops of `number` and `repeat` trials) and how taking the minimum trial filters out background system noise.
* **🐛 The Struggle (Bug/Math Wall):** Figuring out how to pass PyTorch operations to a stopwatch function without Python executing the math beforehand
* **💡 The Fix / Insight:** Used a `lambda` wrapper to delay code execution, turning a finished calculation into a "recipe" that `timeit` can trigger inside its stopwatch. Additionally, realized that CPU operations naturally block and wait, whereas GPU operations require synchronization before stopping the clock to get accurate math timings

### 📅 2026-06-25 | Module 2: Math Operations & Early Measurement Culture
* **🎯 Goal Today:** Understand the difference between in-place and out-of-place operations
* **✅ Accomplished:** Understood the differences in how these operations are carried out in memory
* **🐛 The Struggle (Bug/Math Wall):** None
* **💡 The Fix / Insight:** In-place operations end with an underscore(_) in pytorch. they create new blocks of memory to store new results, Out-of-place operations mutate the existing blocks of memory, replacing the new old results with the new ones. This makes them more efficient