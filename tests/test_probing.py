import numpy as np
import torch

from connect4.analysis import board_labels
from connect4.generate import generate
from connect4.model import GPT, ModelConfig
from connect4.probing import Standardizer, predict, residual_stream, train_linear_probe
from connect4.tokenizer import encode


def test_residual_stream_matches_forward_and_masks_rows():
    torch.manual_seed(0)
    model = GPT(ModelConfig(n_layers=2, d_model=32, n_heads=4)).eval()
    games = generate(10, seed=0)
    tokens = torch.from_numpy(encode(games, end_token=False)[:, :-1]).long()
    valid = torch.from_numpy(board_labels(games).valid)

    full = residual_stream(model, tokens, batch_size=4)
    masked = residual_stream(model, tokens, mask=valid, batch_size=4)
    _, resid = model(tokens, return_resid=True)

    assert len(full) == 3
    for f, m, r in zip(full, masked, resid):
        assert torch.allclose(f, r, atol=1e-5)
        assert m.shape == (int(valid.sum()), 32) and torch.allclose(m, r[valid], atol=1e-5)


def test_linear_probe_learns_a_linear_rule():
    torch.manual_seed(0)
    x = torch.randn(4000, 8)
    y = torch.stack([(x[:, 0] > 0), (x[:, 1] + x[:, 2] > 0)], dim=1).long()  # 2 targets, 2 classes
    norm = Standardizer(x[:3000])
    probe = train_linear_probe(norm(x[:3000]), y[:3000], n_classes=2, epochs=20, lr=1e-2)
    acc = (predict(probe, norm(x[3000:]), n_classes=2) == y[3000:]).float().mean()
    assert acc > 0.97
    assert np.isclose(norm(x[:3000]).mean().item(), 0, atol=1e-5)
