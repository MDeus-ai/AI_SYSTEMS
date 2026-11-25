#include "bench_timer.hpp"

#include <bit>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

void inspect_ieee754_float(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 31) & 0x1U;
    const std::uint32_t exponent = (bits >> 23) & 0xFFU;
    const std::uint32_t mantissa = bits & 0x7FFFFFU;

    std::cout << "Value: " << value
              << " | Sign: " << sign
              << " | Biased Exp: " << exponent << " (2^" << (static_cast<int>(exponent) - 127) << ")"
              << " | Mantissa Hex: 0x" << std::hex << mantissa << std::dec << '\n';
}

} // namespace

int main() {
    std::cout << "=== AI Systems Phase 03: Hello Silicon ===\n\n";

    // 1. Inspect raw IEEE-754 float32 layout (connecting Phase 00 to C++20 hardware bits)
    std::cout << "[1] IEEE-754 FP32 Bit Anatomy:\n";
    inspect_ieee754_float(1.0f);
    inspect_ieee754_float(-0.15625f);
    inspect_ieee754_float(65504.0f); // Max normal FP16 value!

    // 2. Compare Stack pointer vs Heap buffer address
    const float stack_learning_rate = 3e-4f;
    constexpr std::size_t n_elements = 16'000'000; // 64 MB contiguous float32 tensor
    std::vector<float> heap_tensor(n_elements, 1.0001f);

    std::cout << "\n[2] Physical Memory Coordinates:\n";
    std::cout << "Stack scalar (&stack_learning_rate): " << static_cast<const void*>(&stack_learning_rate) << '\n';
    std::cout << "Heap vector header (&heap_tensor):   " << static_cast<const void*>(&heap_tensor) << '\n';
    std::cout << "Heap contiguous buffer (data()):     " << static_cast<const void*>(heap_tensor.data())
              << " (" << (n_elements * sizeof(float)) / (1024 * 1024) << " MiB)\n\n";

    // 3. Measure contiguous triad/scale bandwidth using your reusable C++ timer
    std::cout << "[3] Baseline Contiguous Memory Bandwidth Check:\n";
    const std::size_t bytes_moved = n_elements * sizeof(float) * 2; // 1 read + 1 write per float
    const std::size_t flops = n_elements;                           // 1 multiply per float

    ai_sys::measure_speed_cpp(
        "Contiguous In-Place Scale",
        [&]() {
            float* ptr = heap_tensor.data();
            for (std::size_t i = 0; i < n_elements; ++i) {
                ptr[i] *= 0.999999f;
            }
            ai_sys::do_not_optimize(ptr[0]);
        },
        bytes_moved,
        flops,
        /*warmup_runs=*/5,
        /*repeats=*/5,
        /*loops_per_repeat=*/20
    );

    return 0;
}
