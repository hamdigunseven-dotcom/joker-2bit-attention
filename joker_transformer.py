import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Straight-Through Estimator (STE) Joker Quantizer
def ste_round(x):
    return (torch.round(x) - x).detach() + x

def ste_clamp(x, min_val, max_val):
    return (torch.clamp(x, min_val, max_val) - x).detach() + x

def quantize_joker(W):
    W_centered = W - W.mean()
    gamma = W_centered.abs().mean() + 1e-8
    W_scaled = W_centered / gamma
    # Joker Shift: {-1, 0, 1, 2}
    W_q = ste_clamp(ste_round(W_scaled), -1.0, 2.0)
    return W_q * gamma

# 2. Joker Kuantize Linear Katmanı
class JokerLinear(nn.Linear):
    def forward(self, x):
        w_q = quantize_joker(self.weight)
        return F.linear(x, w_q, self.bias)

# 3. Mini Joker Causal Self-Attention Katmanı
class JokerCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        # Q, K, V Projeksiyonları Joker2-bit
        self.c_attn = JokerLinear(n_embd, 3 * n_embd)
        self.c_proj = JokerLinear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

# 4. Joker Transformer Blok
class JokerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = JokerCausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        # Feed-Forward Network (FFN) Joker2-bit
        self.mlp = nn.Sequential(
            JokerLinear(n_embd, 4 * n_embd),
            nn.GELU(),
            JokerLinear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# 5. Mini Joker-GPT Modeli
class JokerGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=32):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[JokerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = JokerLinear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 6. Eğitim Döngüsü
if __name__ == "__main__":
    text = "joker 2-bit quantization architecture zero multiply inference transformer model python cpp"
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    
    model = JokerGPT(vocab_size=vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Joker-GPT Mimarisi Eğitiliyor...\n")
    model.train()
    for iter in range(300):
        # Mini-batch hazırlığı
        ix = torch.randint(len(data) - 16, (8,))
        x = torch.stack([data[i:i+16] for i in ix])
        y = torch.stack([data[i+1:i+17] for i in ix])

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % 50 == 0:
            print(f"Adım {iter:<4} | Loss: {loss.item():.4f}")

    # Metin Üretim Testi
    model.eval()
    context = torch.tensor([encode("joker ")], dtype=torch.long)
    generated = model.generate(context, max_new_tokens=30)[0].tolist()
    print("\n--- JOKER 2-BIT TRANSFORMER METİN ÜRETİMİ ---")
    print(decode(generated))
