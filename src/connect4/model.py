"""A minimal GPT (decoder-only transformer), sized like Othello-GPT by default."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from connect4.tokenizer import SEQ_LEN, VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    context_len: int = SEQ_LEN - 1  # the last token is only ever a target, never an input
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # (batch, seq, d) -> (batch, heads, seq, d_head): each head attends independently.
        q, k, v = (t.view(batch, seq, self.n_heads, -1).transpose(1, 2) for t in (q, k, v))
        # is_causal: position i may only look at positions <= i, so it can't peek at future moves.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(batch, seq, d))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each sublayer reads the residual stream and adds its output back to it.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self, tokens: torch.Tensor, return_resid: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """tokens (batch, seq) -> logits (batch, seq, vocab).

        With return_resid=True, also returns the residual stream after the embedding
        and after each block (n_layers + 1 tensors of shape (batch, seq, d_model)).
        """
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(positions)
        resid = [x]
        for block in self.blocks:
            x = block(x)
            resid.append(x)
        logits = self.unembed(self.ln_f(x))
        return (logits, resid) if return_resid else logits
