from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tinyllm.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_proj(attn)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int, dropout: float):
        super().__init__()
        hidden_dim = d_model * mlp_ratio
        self.fc = nn.Linear(d_model, hidden_dim)
        self.proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.sequence_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.sequence_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model context {self.config.sequence_length}."
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.sequence_length :]
            logits = self(context)[:, -1, :]
            logits = logits / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return input_ids


def causal_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction="none",
    )
    if loss_mask is None:
        return losses.mean()

    weights = loss_mask.reshape(-1).to(losses.dtype)
    if weights.numel() != losses.numel():
        raise ValueError("loss_mask shape must match targets shape.")
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def estimate_parameters(config: ModelConfig, vocab_size: int) -> int:
    embeddings = vocab_size * config.d_model + config.sequence_length * config.d_model
    block_multiplier = 12 * config.d_model * config.d_model
    return embeddings + config.n_layers * block_multiplier + config.d_model * vocab_size


def model_size_millions(config: ModelConfig, vocab_size: int) -> float:
    return round(estimate_parameters(config, vocab_size) / 1_000_000, 2)
