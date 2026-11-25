#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace ai_sys {

/**
 * Prevents the C++ compiler (-O3 / /O2) from optimizing away a computed value
 * during microbenchmarks when the result is not printed.
 */
template <typename T>
inline void do_not_optimize(T const& value) {
#if defined(_MSC_VER)
    volatile auto sink = &value;
    (void)sink;
#else
    asm volatile("" : : "r,m"(value) : "memory");
#endif
}

struct BenchResult {
    double best_ms{};
    double gbps{};
    double gflops{};
};

/**
 * C++20 twin of your Phase 02 Python `measure_speed` benchmark utility!
 * 1. Runs warmup iterations to prime CPU caches & branch predictors.
 * 2. Runs `repeats` outer trials of `loops_per_repeat` inner calls.
 * 3. Takes the minimum trial (`min`) to filter out OS context-switch noise.
 */
template <typename Func>
BenchResult measure_speed_cpp(
    std::string_view label,
    Func&& fn,
    std::size_t bytes_moved = 0,
    std::size_t flops = 0,
    int warmup_runs = 10,
    int repeats = 5,
    int loops_per_repeat = 50
) {
    for (int i = 0; i < warmup_runs; ++i) {
        fn();
    }

    std::vector<double> trial_seconds;
    trial_seconds.reserve(static_cast<std::size_t>(repeats));

    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        for (int l = 0; l < loops_per_repeat; ++l) {
            fn();
        }
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        trial_seconds.push_back(elapsed.count());
    }

    const double best_total_sec = *std::min_element(trial_seconds.begin(), trial_seconds.end());
    const double sec_per_run = best_total_sec / static_cast<double>(loops_per_repeat);
    const double ms_per_run = sec_per_run * 1000.0;

    const double gbps = (bytes_moved > 0)
        ? (static_cast<double>(bytes_moved) / 1e9) / sec_per_run
        : 0.0;
    const double gflops = (flops > 0)
        ? (static_cast<double>(flops) / 1e9) / sec_per_run
        : 0.0;

    std::cout << std::fixed << std::setprecision(4)
              << "[BENCH] " << std::left << std::setw(28) << label
              << " | " << std::right << std::setw(9) << ms_per_run << " ms";
    if (bytes_moved > 0) {
        std::cout << " | " << std::setw(8) << std::setprecision(2) << gbps << " GB/s";
    }
    if (flops > 0) {
        std::cout << " | " << std::setw(8) << std::setprecision(2) << gflops << " GFLOP/s";
    }
    std::cout << '\n';

    return BenchResult{ms_per_run, gbps, gflops};
}

} // namespace ai_sys
