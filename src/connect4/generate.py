"""Generate games of uniformly random legal moves and save them as a padded array.

Usage:
    uv run python -m connect4.generate --n 1000000 --seed 0 --out data/games.npy
"""

import argparse
import random
from pathlib import Path

import numpy as np

from connect4.game import COLS, ROWS, Game

MAX_MOVES = ROWS * COLS
PAD = -1  # fills the unused tail of shorter games; never a valid column


def random_game(rng: random.Random) -> list[int]:
    """Play one game where each move is picked uniformly from the legal columns."""
    game = Game()
    while not game.is_over:
        game.play(rng.choice(game.legal_moves()))
    return game.moves


def generate(n: int, seed: int) -> np.ndarray:
    """Return an (n, 42) int8 array: row i holds game i's columns, then PAD to the end."""
    rng = random.Random(seed)
    games = np.full((n, MAX_MOVES), PAD, dtype=np.int8)
    for i in range(n):
        moves = random_game(rng)
        games[i, : len(moves)] = moves
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, required=True, help="number of games")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    games = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, games)
    print(f"saved {args.n} games to {args.out} ({games.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
