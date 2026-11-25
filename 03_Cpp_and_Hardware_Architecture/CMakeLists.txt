cmake_minimum_required(VERSION 3.20)
project(AI_Systems_Phase03_Engine LANGUAGES CXX)

# Enforce Modern C++20 across all targets
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release build if not specified so benchmarks never run unoptimized by accident!
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Optional OpenMP support for Module 3.6 & Boss Fight multithreading
find_package(OpenMP QUIET)
option(ENABLE_ASAN "Enable AddressSanitizer (/fsanitize=address or -fsanitize=address)" OFF)

# Reusable compiler flags interface target (strict warnings + native SIMD in Release)
add_library(systems_flags INTERFACE)
target_include_directories(systems_flags INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
    target_compile_options(systems_flags INTERFACE
        /W4 /permissive-
        $<$<CONFIG:Release>:/O2 /arch:AVX2 /fp:fast>
        $<$<BOOL:${ENABLE_ASAN}>:/fsanitize=address>
    )
else()
    target_compile_options(systems_flags INTERFACE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
        $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fno-omit-frame-pointer>
    )
    if(ENABLE_ASAN)
        target_link_options(systems_flags INTERFACE -fsanitize=address)
    endif()
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(systems_flags INTERFACE OpenMP::OpenMP_CXX)
endif()

# Starter Executable: Inspect IEEE-754 bits, memory alignment & benchmark harness
add_executable(hello_silicon src/hello_silicon.cpp)
target_link_libraries(hello_silicon PRIVATE systems_flags)
