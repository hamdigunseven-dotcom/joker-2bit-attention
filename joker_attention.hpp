#ifndef JOKER_ATTENTION_HPP
#define JOKER_ATTENTION_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <immintrin.h> // AVX2 & FMA SIMD

namespace joker {

// 2-Bit Paketli Katman Yapısı
struct PackedLinear {
    int in_features;
    int out_features;
    float gamma;
    std::vector<uint8_t> packed_weights;
    std::vector<float> bias;

    void init(int in_f, int out_f, float scale = 0.05f) {
        in_features = in_f;
        out_features = out_f;
        gamma = scale;
        int total_weights = in_f * out_f;
        int packed_bytes = (total_weights + 3) / 4;
        packed_weights.assign(packed_bytes, (3 << 6) | (2 << 4) | (1 << 2) | 0); // Test Deseni
        bias.assign(out_f, 0.01f);
    }
};

// Modüler AVX2 Zero-Multiply Attention Sınıfı
class SelfAttentionAVX2 {
private:
    int n_embd;
    int n_head;
    int head_dim;

    PackedLinear W_q;
    PackedLinear W_k;
    PackedLinear W_v;

    // AVX2 Vektörel Zero-Multiply Projeksiyon Motoru
    void forwardLinearAVX2(const PackedLinear& layer, const float* input, float* output) const {
        for (int r = 0; r < layer.out_features; ++r) {
            int row_offset = r * layer.in_features;
            __m256 v_acc = _mm256_setzero_ps();

            int c = 0;
            // 8 Float Vektörel Döngü
            for (; c <= layer.in_features - 8; c += 8) {
                __m256 v_x = _mm256_loadu_ps(&input[c]);

                int flat_idx = row_offset + c;
                int packed_idx = flat_idx / 4;
                uint16_t packed_16 = *reinterpret_cast<const uint16_t*>(&layer.packed_weights[packed_idx]);

                alignas(32) float weights_unpacked[8];
                for (int i = 0; i < 8; ++i) {
                    uint8_t w = (packed_16 >> (i * 2)) & 0x03;
                    switch (w) {
                        case 0: weights_unpacked[i] = -1.0f; break; // -1
                        case 1: weights_unpacked[i] =  0.0f; break; //  0 (NOP)
                        case 2: weights_unpacked[i] =  1.0f; break; // +1
                        case 3: weights_unpacked[i] =  2.0f; break; // JOKER (2x Shift)
                    }
                }

                __m256 v_w = _mm256_load_ps(weights_unpacked);
                v_acc = _mm256_fmadd_ps(v_x, v_w, v_acc);
            }

            alignas(32) float acc_array[8];
            _mm256_store_ps(acc_array, v_acc);
            float final_acc = 0.0f;
            for (int i = 0; i < 8; ++i) final_acc += acc_array[i];

            // Kuyruk Elemanları
            for (; c < layer.in_features; ++c) {
                int flat_idx = row_offset + c;
                uint8_t packed_byte = layer.packed_weights[flat_idx / 4];
                uint8_t w = (packed_byte >> ((c % 4) * 2)) & 0x03;
                float x = input[c];
                switch (w) {
                    case 0: final_acc -= x; break;
                    case 2: final_acc += x; break;
                    case 3: final_acc += (x + x); break;
                }
            }

            output[r] = (final_acc * layer.gamma) + layer.bias[r];
        }
    }

public:
    SelfAttentionAVX2(int embed_dim, int num_heads) 
        : n_embd(embed_dim), n_head(num_heads) {
        head_dim = n_embd / n_head;
        W_q.init(n_embd, n_embd);
        W_k.init(n_embd, n_embd);
        W_v.init(n_embd, n_embd);
    }

    std::vector<float> forward(const std::vector<float>& input_seq, int seq_len) const {
        std::vector<float> Q(seq_len * n_embd);
        std::vector<float> K(seq_len * n_embd);
        std::vector<float> V(seq_len * n_embd);

        for (int t = 0; t < seq_len; ++t) {
            const float* x_t = &input_seq[t * n_embd];
            forwardLinearAVX2(W_q, x_t, &Q[t * n_embd]);
            forwardLinearAVX2(W_k, x_t, &K[t * n_embd]);
            forwardLinearAVX2(W_v, x_t, &V[t * n_embd]);
        }

        std::vector<float> att_out(seq_len * n_embd, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int h = 0; h < n_head; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                std::vector<float> scores(seq_len, -1e9f);
                float max_score = -1e9f;

                for (int j = 0; j <= i; ++j) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += Q[i * n_embd + h * head_dim + d] * K[j * n_embd + h * head_dim + d];
                    }
                    score *= scale;
                    scores[j] = score;
                    if (score > max_score) max_score = score;
                }

                float exp_sum = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    exp_sum += scores[j];
                }
                for (int j = 0; j <= i; ++j) scores[j] /= exp_sum;

                for (int d = 0; d < head_dim; ++d) {
                    float v_acc = 0.0f;
                    for (int j = 0; j <= i; ++j) {
                        v_acc += scores[j] * V[j * n_embd + h * head_dim + d];
                    }
                    att_out[i * n_embd + h * head_dim + d] = v_acc;
                }
            }
        }

        return att_out;
    }
};

} // namespace joker

#endif // JOKER_ATTENTION_HPP
