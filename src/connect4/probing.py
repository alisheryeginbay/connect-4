"""Reusable pieces for linear probing: residual-stream extraction, standardization, probe training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from connect4.model import GPT


def residual_stream(model: GPT, tokens: torch.Tensor, mask: torch.Tensor | None = None,
                    batch_size: int = 512) -> list[torch.Tensor]:
    """Residual stream after the embedding and after each block (n_layers + 1 tensors), on the CPU.

    tokens: (games, positions) on the model's device. With mask (games, positions) bool, only those
    positions are kept and each tensor is (rows, d_model); otherwise (games, positions, d_model).
    Masking per batch keeps memory low: the full stream for every layer never exists at once.
    """
    out: list[list[torch.Tensor]] = [[] for _ in range(model.cfg.n_layers + 1)]
    with torch.no_grad():
        for b in range(0, len(tokens), batch_size):
            _, resid = model(tokens[b : b + batch_size], return_resid=True)
            keep = None if mask is None else mask[b : b + batch_size]
            for layer, r in enumerate(resid):
                r = r.cpu()
                out[layer].append(r if keep is None else r[keep])
    return [torch.cat(chunks) for chunks in out]


class Standardizer:
    """Per-feature mean/std from training rows, applied unchanged to any other rows."""

    def __init__(self, train_x: torch.Tensor) -> None:
        self.mean = train_x.mean(0)
        self.std = train_x.std(0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + 1e-6)

    def state(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}


def probe_logits(probe: nn.Linear, x: torch.Tensor, n_classes: int) -> torch.Tensor:
    """(rows, d) -> (rows, n_classes, n_targets). Outputs are class-major: index = cls * n_targets + target."""
    return rearrange(probe(x), "n (cls target) -> n cls target", cls=n_classes)


def train_linear_probe(x: torch.Tensor, y: torch.Tensor, n_classes: int, *, epochs: int = 50,
                       lr: float = 1e-2, batch_size: int = 512, device: str = "cpu",
                       seed: int = 0) -> nn.Linear:
    """Fit one linear map predicting n_targets labels (each in 0..n_classes-1) at once.

    x: (rows, d) standardized activations; y: (rows, n_targets) int labels.
    """
    torch.manual_seed(seed)
    n_targets = y.shape[1]
    probe = nn.Linear(x.shape[1], n_classes * n_targets).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    x, y = x.to(device), y.long().to(device)
    for _ in range(epochs):
        order = torch.randperm(len(x), device=device)
        for b in range(0, len(x), batch_size):
            idx = order[b : b + batch_size]
            loss = F.cross_entropy(probe_logits(probe, x[idx], n_classes), y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return probe


@torch.no_grad()
def predict(probe: nn.Linear, x: torch.Tensor, n_classes: int) -> torch.Tensor:
    """(rows, d) -> (rows, n_targets) predicted class, on the CPU."""
    device = probe.weight.device
    return probe_logits(probe, x.to(device), n_classes).argmax(dim=1).cpu()
