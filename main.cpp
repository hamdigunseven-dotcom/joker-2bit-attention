#include <iostream>
#include "joker_attention.hpp" // Header-Only Kütüphanemiz

int main() {
    int seq_len = 16;
    int n_embd = 64;
    int n_head = 4;

    // Namespace üzerinden çağırım
    joker::SelfAttentionAVX2 engine(n_embd, n_head);
    std::vector<float> input_seq(seq_len * n_embd, 0.25f);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> output = engine.forward(input_seq, seq_len);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> duration = end - start;

    std::cout << "[JOKER_ATTENTION.HPP] Cikarim Tamamlandi!" << std::endl;
    std::cout << "Süre: " << duration.count() << " us" << std::endl;

    return 0;
}
