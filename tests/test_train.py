import numpy as np
import torch

from connect4.generate import PAD as MOVE_PAD
from connect4.model import GPT, ModelConfig
from connect4.tokenizer import END, VOCAB_SIZE, encode
from connect4.train import legal_mask


def test_legal_mask_on_a_vertical_win():
    games = np.full((1, 42), MOVE_PAD, dtype=np.int8)
    games[0, :7] = [0, 1, 0, 1, 0, 1, 0]  # first player wins in column 0 on move 7
    for end_token in (False, True):
        mask = legal_mask(games, end_token)[0]
        assert mask[0, :7].all() and not mask[0, 7:].any()  # at BOS: all 7 columns
        assert not mask[7, :7].any()  # after the win: no column is legal
        assert mask[7, END] == end_token
        assert not mask[8:].any()


def test_model_shapes():
    cfg = ModelConfig(n_layers=2, d_model=32, n_heads=4)
    tokens = torch.from_numpy(encode(np.zeros((3, 42), dtype=np.int8), end_token=True)).long()[:, :-1]
    logits, resid = GPT(cfg)(tokens, return_resid=True)
    assert logits.shape == (3, cfg.context_len, VOCAB_SIZE)
    assert len(resid) == cfg.n_layers + 1
    assert all(r.shape == (3, cfg.context_len, cfg.d_model) for r in resid)
