"""Train the GPT to predict the next token of random Connect 4 games.

Usage (local smoke test):
    uv run python -m connect4.generate --n 100000 --out data/games.npy
    uv run python -m connect4.train --data data/games.npy --out runs/smoke --epochs 1 --small
"""

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from connect4.game import Game
from connect4.generate import PAD as MOVE_PAD
from connect4.model import GPT, ModelConfig
from connect4.tokenizer import END, PAD, VOCAB_SIZE, encode


@dataclass
class TrainConfig:
    data: str
    out: str
    end_token: bool = False
    epochs: int = 2
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    val_games: int = 10_000
    eval_every: int = 1000
    log_every: int = 100
    seed: int = 0


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def legal_mask(games: np.ndarray, end_token: bool) -> np.ndarray:
    """(n, 42) games -> (n, 43, vocab) bool: which tokens are legal next tokens at each input position.

    Position t has seen BOS plus the first t moves. Positions after the game are all-False.
    """
    n = len(games)
    mask = np.zeros((n, games.shape[1] + 1, VOCAB_SIZE), dtype=bool)
    for i, row in enumerate(games):
        game = Game()
        for t, col in enumerate(row[row != MOVE_PAD].tolist() + [None]):
            mask[i, t, game.legal_moves()] = True
            if col is None:
                mask[i, t, END] = end_token
                break
            game.play(col)
    return mask


def loss_floor(tokens: torch.Tensor, legal: torch.Tensor) -> float:
    """The best achievable loss: games are uniform over legal moves, so it's the mean ln(#legal)."""
    real = tokens[:, 1:] != PAD
    return legal.sum(-1)[real].float().log().mean().item()


@torch.no_grad()
def evaluate(model: GPT, tokens: torch.Tensor, legal: torch.Tensor, batch_size: int) -> dict[str, float]:
    """Validation loss, and how often the model's top prediction is a legal move (the paper's metric)."""
    model.eval()
    loss_sum, n_targets, n_legal = 0.0, 0, 0
    for i in range(0, len(tokens), batch_size):
        x, y = tokens[i : i + batch_size, :-1], tokens[i : i + batch_size, 1:]
        logits = model(x).float()
        loss_sum += F.cross_entropy(logits.transpose(1, 2), y, ignore_index=PAD, reduction="sum").item()
        real = y != PAD
        top = logits.argmax(-1, keepdim=True)
        n_legal += legal[i : i + batch_size].gather(-1, top).squeeze(-1)[real].sum().item()
        n_targets += real.sum().item()
    model.train()
    return {"val_loss": loss_sum / n_targets, "legal_top1": n_legal / n_targets}


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay down to 10% of the peak learning rate."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(cfg: TrainConfig, model_cfg: ModelConfig = ModelConfig()) -> None:
    torch.manual_seed(cfg.seed)
    device = pick_device()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    games = np.load(cfg.data)
    val_games, train_games = games[: cfg.val_games], games[cfg.val_games :]
    # The whole dataset fits on the GPU (5M games is ~220 MB), so batches are just index slices.
    train_tokens = torch.from_numpy(encode(train_games, cfg.end_token)).long().to(device)
    val_tokens = torch.from_numpy(encode(val_games, cfg.end_token)).long().to(device)
    val_legal = torch.from_numpy(legal_mask(val_games, cfg.end_token)).to(device)

    model = GPT(model_cfg).to(device)
    print(f"device {device}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"{len(train_tokens)} train / {len(val_tokens)} val games, "
          f"val loss floor {loss_floor(val_tokens, val_legal):.4f}")

    # Weight decay on matrices only; biases, LayerNorm and embeddings are usually left alone.
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "emb" in n]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )

    steps_per_epoch = len(train_tokens) // cfg.batch_size
    total_steps = cfg.epochs * steps_per_epoch
    # bfloat16 autocast roughly doubles speed on modern GPUs; elsewhere run in plain float32.
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    history = []

    def save(metrics: dict) -> None:
        history.append(metrics)
        torch.save({"model": model.state_dict(), "model_cfg": dataclasses.asdict(model_cfg),
                    "train_cfg": dataclasses.asdict(cfg), "step": step}, out / "model.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))

    step, start = 0, time.time()
    for epoch in range(cfg.epochs):
        order = torch.randperm(len(train_tokens), device=device)
        for b in range(steps_per_epoch):
            batch = train_tokens[order[b * cfg.batch_size : (b + 1) * cfg.batch_size]]
            # Input is every token but the last; target is every token but the first (shift by one).
            x, y = batch[:, :-1], batch[:, 1:]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, cfg)
            with autocast:
                logits = model(x)
            loss = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=PAD)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                      f"lr {group['lr']:.2e} {time.time() - start:.0f}s")
            if step % cfg.eval_every == 0 or step == total_steps:
                with autocast:
                    metrics = evaluate(model, val_tokens, val_legal, cfg.batch_size)
                metrics.update(step=step, train_loss=loss.item())
                print(f"  eval step {step}: val_loss {metrics['val_loss']:.4f} "
                      f"legal_top1 {metrics['legal_top1']:.4%}")
                save(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--end-token", action="store_true")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--small", action="store_true", help="2-layer, 128-dim model for quick local checks")
    args = parser.parse_args()

    cfg = TrainConfig(data=args.data, out=args.out, end_token=args.end_token,
                      epochs=args.epochs, batch_size=args.batch_size)
    model_cfg = ModelConfig(n_layers=2, d_model=128, n_heads=4) if args.small else ModelConfig()
    train(cfg, model_cfg)


if __name__ == "__main__":
    main()
