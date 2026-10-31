import numpy as np
import torch

from connect4.analysis import board_labels, load_model
from connect4.game import FIRST, SECOND
from connect4.generate import PAD as MOVE_PAD
from connect4.generate import generate
from connect4.model import GPT, ModelConfig
from connect4.tokenizer import encode


def test_board_labels_on_a_vertical_win():
    games = np.full((1, 42), MOVE_PAD, dtype=np.int8)
    games[0, :7] = [0, 1, 0, 1, 0, 1, 0]  # first player wins in column 0 on move 7
    labels = board_labels(games)

    assert not labels.board[0, 0].any() and not labels.heights[0, 0].any()  # at BOS: empty board
    assert labels.heights[0, 2].tolist() == [1, 1, 0, 0, 0, 0, 0]  # after moves 0, 1
    assert labels.board[0, 2, 0, 0] == FIRST and labels.board[0, 2, 0, 1] == SECOND
    assert labels.player[0, :3].tolist() == [FIRST, SECOND, FIRST]
    assert labels.heights[0, 7].tolist() == [4, 3, 0, 0, 0, 0, 0]  # final board, after the win
    assert labels.valid[0, :8].all() and not labels.valid[0, 8:].any()
    # At position 2 it's the first player's turn, so their piece in column 0 is "mine" (+1).
    assert labels.relative_board[0, 2, 0, 0] == 1 and labels.relative_board[0, 1, 0, 0] == -1


def test_load_model_roundtrip(tmp_path):
    cfg = ModelConfig(n_layers=2, d_model=32, n_heads=4)
    model = GPT(cfg)
    torch.save({"model": model.state_dict(), "model_cfg": cfg.__dict__, "train_cfg": {"end_token": True}},
               tmp_path / "model.pt")
    loaded, train_cfg = load_model(tmp_path / "model.pt")
    tokens = torch.from_numpy(encode(generate(4, seed=0), end_token=True)).long()[:, :-1]
    assert train_cfg["end_token"] is True
    assert torch.allclose(loaded(tokens), model.eval()(tokens))
