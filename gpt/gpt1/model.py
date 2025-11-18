import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)
from config import *


class Head(nn.Module):
    def __init__(self, head_size: int = N_EMBED) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        wei = query @ key.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        final_out = wei @ value
        return final_out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int = N_EMBED) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class MultiHeadAttentionBatched(nn.Module):
    def __init__(self, n_embed: int, num_heads: int = N_EMBED) -> None:
        super().__init__()
        self.num_head = num_heads
        self.head_size = n_embed // num_heads
        if n_embed % num_heads != 0:
            raise ValueError("Embedding size must be divisible by number of heads")
        self.qkv_proj = nn.Linear(N_EMBED, 3 * N_EMBED)
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer(
            "tril", torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH).unsqueeze(0).unsqueeze(0))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        query, key, value = qkv.chunk(3, -1)
        # B, head_size, T, num_head
        query = query.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        # B, head_size, T, num_head
        key = key.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        # B, head_size, T, num_head
        value = value.view(B, T, self.num_head, self.head_size).transpose(1, 2)
        attn = query @ key.transpose(-2, -1) * self.head_size**-0.5
        attn = attn.masked_fill(self.tril == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = attn @ value
        output = output.transpose(1, 2).contiguous().view(B, T, self.num_head * self.head_size)
        output = self.proj(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, n_embed: int = N_EMBED) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttentionBatched(n_embed, n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(self.vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(CONTEXT_LENGTH, N_EMBED)
        self.blocks = nn.Sequential(
            *[Block(N_EMBED, n_head=4) for _ in range(N_LAYER)], nn.LayerNorm(N_EMBED)
        )
        self.lm_head = nn.Linear(N_EMBED, self.vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        token_embed = self.token_embedding_table(idx)  # (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_embed + pos_embed
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B,T, vocal_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONTEXT_LENGTH:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
