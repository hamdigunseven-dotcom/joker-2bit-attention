# Joker 2-Bit Zero-Multiply Inference Architecture for Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language: C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![SIMD: AVX2](https://img.shields.io/badge/SIMD-AVX2%2FFMA-green.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)

A zero-multiply, AVX2-accelerated 2-bit Self-Attention C++ inference engine and PyTorch Quantization-Aware Training (QAT) framework for Transformer architectures.

---

## Executive Summary

Standard neural network inference relies heavily on Floating-Point Units (FPUs) to execute dense General Matrix-Vector Multiplication (GEMV). Modern ternary quantization schemes ($\{-1, 0, 1\}$) leave the 4th state of 2-bit representations ($2^2 = 4$) unmapped. 

The **Joker 2-Bit Architecture** maps the unused binary state (`11`) to a $2\times$ power-of-two shift scalar (the "Joker"). By pairing Quantization-Aware Training (QAT) with a C++ AVX2 inference kernel, this architecture completely eliminates hardware multiplier instructions (`MUL`) for projection layers, compresses weight representations by **~4x**, and achieves sub-millisecond execution times for Transformer Self-Attention projections.

---

## Key Features & Architecture

* **Zero Hardware Multipliers:** Replaces FPU multiplication steps in matrix projections with bitwise additions, subtractions, and bit-shifts.
* **4x Weight Compression:** Packs 4 distinct 2-bit Joker parameters into a single `uint8_t` byte, reducing model footprint from $231\text{ KB}$ to $59\text{ KB}$ ($\sim 15.6\times$ vs Float32).
* **Header-Only Integration:** Low-level inference engine implemented as a dependency-free C++ header (`include/joker_attention.hpp`).
* **AVX2 Vectorization:** Utilizes 256-bit SIMD registers (`__m256`) for parallel in-register unpacking and accumulation.

---

## Experimental Benchmarks

Benchmarked on x86_64 architecture using GCC (`-O3 -mavx2 -mfma`):

### 1. Model Memory Compression
| Format / Precision | File Size | Compression Ratio | Memory Reduction |
| :--- | :--- | :--- | :--- |
| **Float32 Baseline** | $924\text{ KB}$ | $1.00\times$ | $0.0\%$ |
| **Int8 Unpacked** | $231\text{ KB}$ | $4.00\times$ | $75.0\%$ |
| **Joker 2-Bit Packed** | **$59\text{ KB}$** | **$15.66\times$** | **$93.6\%$** |

### 2. Execution Latency
| Target Layer / Kernel | Execution Time ($\mu\text{s}$) | Hardware Multiplier Status |
| :--- | :--- | :--- |
| **Scalar Int8 (MLP Layer)** | $1087\ \mu\text{s}$ ($1.08\text{ ms}$) | Bypassed |
| **AVX2 SIMD 2-Bit (MLP Layer)** | $1103\ \mu\text{s}$ ($1.10\text{ ms}$) | Vectorized |
| **Native Self-Attention ($Q, K, V$)** | **$156\ \mu\text{s}$ ($0.156\text{ ms}$)** | **Zero-Multiply** |
| **Header-Only Engine (`joker_attention.hpp`)** | **$319\ \mu\text{s}$ ($0.319\text{ ms}$)** | **Zero-Multiply** |

---

## Quick Start

### C++ Inference Engine

```cpp
#include <iostream>
#include "joker_attention.hpp"

int main() {
    int seq_len = 16;
    int n_embd = 64;
    int n_head = 4;

    joker::SelfAttentionAVX2 engine(n_embd, n_head);
    std::vector<float> input_seq(seq_len * n_embd, 0.25f);

    std::vector<float> output = engine.forward(input_seq, seq_len);
    std::cout << "Inference completed successfully!" << std::endl;
    return 0;
}
