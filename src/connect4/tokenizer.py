"""Turn games (rows of columns padded with -1) into fixed-length token sequences.

A sequence looks like:  BOS m1 m2 ... mk [END] PAD PAD ...
Move tokens are the column numbers themselves (0-6), so no lookup table is needed.
"""

import numpy as np

from connect4.game import COLS
from connect4.generate import MAX_MOVES
from connect4.generate import PAD as MOVE_PAD

BOS = COLS  # 7: start of game, so the first move has something to be predicted from
END = COLS + 1  # 8: game over (someone won, or the board is full)
PAD = COLS + 2  # 9: filler after the game; its predictions should be ignored by the loss
VOCAB_SIZE = COLS + 3
SEQ_LEN = 1 + MAX_MOVES + 1  # BOS + up to 42 moves + END

TOKEN_NAMES = [str(c) for c in range(COLS)] + ["BOS", "END", "PAD"]


def encode(games: np.ndarray, end_token: bool) -> np.ndarray:
    """(n, 42) games -> (n, 44) int8 token sequences.

    With end_token=False no END is written and the game is followed directly by PAD.
    The vocabulary and length stay the same either way, so models are comparable.
    """
    n = len(games)
    tokens = np.full((n, SEQ_LEN), PAD, dtype=np.int8)
    tokens[:, 0] = BOS
    tokens[:, 1 : 1 + MAX_MOVES] = np.where(games == MOVE_PAD, PAD, games)
    if end_token:
        lengths = (games != MOVE_PAD).sum(axis=1)
        tokens[np.arange(n), 1 + lengths] = END
    return tokens


def decode(tokens: np.ndarray) -> list[int]:
    """One token sequence -> the list of columns played."""
    moves = []
    for t in tokens[1:]:  # skip BOS
        if t in (END, PAD):
            break
        moves.append(int(t))
    return moves


def to_string(tokens: np.ndarray) -> str:
    """Human-readable form, e.g. 'BOS 3 3 4 END PAD'."""
    return " ".join(TOKEN_NAMES[t] for t in tokens)
