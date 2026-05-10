import torch
import torch.nn.functional as F
import math

# ── Słownik ──────────────────────────────────────────────────────────────────
vocab      = {"Ala": 0, "ma": 1, "kota": 2, "a": 3, "kot": 4, "Alę": 5}
id_to_word = {v: k for k, v in vocab.items()}
V = 6
D = 8

sequence = ["Ala", "ma", "kota", "a", "kot", "ma", "Alę"]
tokens   = torch.tensor([vocab[w] for w in sequence])

# ── Kodowanie pozycyjne (stałe — jak w wersji numpy) ─────────────────────────
def pos_enc(pos):
    p = torch.zeros(D)
    p[6] = math.sin(pos * math.pi / 6)
    p[7] = math.cos(pos * math.pi / 6)
    return p

# ── Parametry — tym razem losowe i z requires_grad=True ──────────────────────
# W wersji numpy: E=jednostkowa, Wq=Wk=Wv=jednostkowe, W_lm=lstsq
# Tu: wszystkie losowe, wszystkie uczone przez gradient descent
torch.manual_seed(42)
E    = torch.randn(V, D) * 0.1
Wq   = torch.randn(D, D) * 0.1
Wk   = torch.randn(D, D) * 0.1
Wv   = torch.randn(D, D) * 0.1
W_lm = torch.randn(D, V) * 0.1

params = [E, Wq, Wk, Wv, W_lm]
for p in params:
    p.requires_grad_(True)

# ── Forward pass — identyczny jak w numpy, tylko tensory PyTorch ──────────────
def forward(prefix_tokens):
    x_seq = [E[t] + pos_enc(i) for i, t in enumerate(prefix_tokens)]

    attn_out = []
    for i, xi in enumerate(x_seq):
        Q       = xi @ Wq
        scores  = torch.stack([Q @ (x_seq[j] @ Wk)
                                for j in range(i + 1)]) / D ** 0.5
        weights = F.softmax(scores, dim=0)
        out     = sum(weights[j] * (x_seq[j] @ Wv) for j in range(i + 1))
        attn_out.append(out)

    hidden = [x + a for x, a in zip(x_seq, attn_out)]
    logits = torch.stack([h @ W_lm for h in hidden])  # (n, V)
    return logits

# ── Trening ───────────────────────────────────────────────────────────────────
# Teacher forcing: wejście = tokens[0..5], cele = tokens[1..6]
# Loss to średnia cross-entropy po wszystkich pozycjach
optimizer = torch.optim.Adam(params, lr=0.05)
prefix    = tokens[:-1]
targets   = tokens[1:]

print("Trening:\n")
for epoch in range(3000):
    optimizer.zero_grad()
    logits = forward(prefix)
    loss   = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0 or epoch == 2999:
        acc = (logits.detach().argmax(dim=1) == targets).float().mean()
        print(f"  epoch {epoch:4d}  loss={loss.item():.4f}  acc={acc:.0%}")

# ── Test ──────────────────────────────────────────────────────────────────────
print("\nPo treningu:\n")
for i in range(1, len(tokens)):
    with torch.no_grad():
        logits = forward(tokens[:i])
        pred   = id_to_word[logits[-1].argmax().item()]
    ctx = " ".join(sequence[:i])
    ok  = "✓" if pred == sequence[i] else "✗"
    print(f"  {ok}  [{ctx}] → {pred}  (oczekiwano: {sequence[i]})")
