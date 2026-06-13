import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from transformer_blocks import Block


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()


block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


def train_model(model_type, vocab_size_setting):
    print("\n" + "=" * 50)
    print(f"TRAINING TOKENIZER: {model_type.upper()}")
    print("=" * 50)

    model_prefix = f"tokenizer_{model_type}"

    spm.SentencePieceTrainer.Train(
        input="corpus.txt",
        model_prefix=model_prefix,
        vocab_size=vocab_size_setting,
        model_type=model_type,
        character_coverage=0.9995
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    ids = sp.encode(text, out_type=int)
    data = torch.tensor(ids, dtype=torch.long)

    vocab_size = sp.get_piece_size()
    print("Vocab size:", vocab_size)
    print("Total token:", len(data))

    def get_batch(batch_size=16):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    model = TinyGPT(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []

    for step in range(epochs):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 300 == 0:
            print(f"Step {step}, loss={loss.item():.4f}")
            losses.append((step, loss.item()))

    context = torch.tensor([sp.encode("machine learning")], dtype=torch.long)
    out = model.generate(context, max_new_tokens=100)

    generated_ids = out[0].tolist()
    generated_text = sp.decode(generated_ids)

    print("\nGenerated text:")
    print(generated_text)

    return {
        "tokenizer": model_type,
        "vocab_size": vocab_size,
        "total_token": len(data),
        "final_loss": losses[-1][1],
        "generated_text": generated_text
    }


if __name__ == "__main__":
    results = []

    tokenizer_modes = [
        ("char", 100),
        ("bpe", 100),
        ("unigram", 100),
    ]

    for mode, vocab in tokenizer_modes:
        result = train_model(mode, vocab)
        results.append(result)

    print("\n" + "=" * 50)
    print("RINGKASAN HASIL")
    print("=" * 50)

    for r in results:
        print(f"Tokenizer     : {r['tokenizer']}")
        print(f"Vocab Size    : {r['vocab_size']}")
        print(f"Total Token   : {r['total_token']}")
        print(f"Final Loss    : {r['final_loss']:.4f}")
        print(f"Generated Text: {r['generated_text'][:200]}")
        print("-" * 50)