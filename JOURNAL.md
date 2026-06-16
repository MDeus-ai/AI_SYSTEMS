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


