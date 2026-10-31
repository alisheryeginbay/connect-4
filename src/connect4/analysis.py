"""Helpers for interpretability experiments: load a trained model, and compute board labels.

Labels line up with the model's input positions: position t has seen BOS plus the
first t moves, so labels[:, t] describe the board after t moves.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from connect4.game import COLS, ROWS, Game
from connect4.generate import MAX_MOVES
from connect4.generate import PAD as MOVE_PAD
from connect4.model import GPT, ModelConfig

N_POSITIONS = MAX_MOVES + 1  # BOS plus up to 42 moves = the model's 43 input positions


def load_model(path: str | Path, device: str = "cpu") -> tuple[GPT, dict]:
    """Rebuild a trained GPT from model.pt. Returns (model in eval mode, its train config).

    The train config says e.g. whether the model was trained with the END token,
    which you need to encode games the same way: encode(games, train_cfg["end_token"]).
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(ModelConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["train_cfg"]


@dataclass
class BoardLabels:
    """Board state at every input position of every game. n = number of games, T = 43."""

    heights: np.ndarray  # (n, T, 7) int8: pieces in each column, 0-6
    board: np.ndarray  # (n, T, 6, 7) int8: +1 first player, -1 second, 0 empty; row 0 is the bottom
    player: np.ndarray  # (n, T) int8: whose turn it is, +1 first / -1 second
    valid: np.ndarray  # (n, T) bool: False after the game ended (inputs there are PAD, labels are 0)

    @property
    def relative_board(self) -> np.ndarray:
        """(n, T, 6, 7): +1 = the player about to move ("mine"), -1 = the opponent ("theirs")."""
        return self.board * self.player[:, :, None, None]


def board_labels(games: np.ndarray) -> BoardLabels:
    """(n, 42) games padded with -1 -> labels for each of the 43 input positions."""
    n = len(games)
    heights = np.zeros((n, N_POSITIONS, COLS), dtype=np.int8)
    board = np.zeros((n, N_POSITIONS, ROWS, COLS), dtype=np.int8)
    player = np.zeros((n, N_POSITIONS), dtype=np.int8)
    valid = np.zeros((n, N_POSITIONS), dtype=bool)
    for i, row in enumerate(games):
        game = Game()
        moves = row[row != MOVE_PAD].tolist()
        for t in range(len(moves) + 1):
            heights[i, t] = game.heights
            board[i, t] = game.board
            player[i, t] = game.player
            valid[i, t] = True
            if t < len(moves):
                game.play(moves[t])
    return BoardLabels(heights, board, player, valid)
