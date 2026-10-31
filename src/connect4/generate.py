"""Generate games of uniformly random legal moves and save them as a padded array.

A dataset is either one .npy file or a directory of shard files, one per seed.
Different seeds can still produce the same short game by chance, so a held-out set
should be generated with --exclude pointing at the training data.

Usage:
    uv run python -m connect4.generate --n 100000 --seed 0 --out data/train/shard_0000.npy
    uv run python -m connect4.generate --n 100000 --seed 1000000 --out data/probe.npy --exclude data/train
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


def shard_name(seed: int) -> str:
    return f"shard_{seed:07d}.npy"


def load_games(path: str | Path) -> np.ndarray:
    """Load one .npy file, or concatenate every shard in a directory (in seed order)."""
    path = Path(path)
    if path.is_dir():
        shards = sorted(path.glob("shard_*.npy"))
        if not shards:
            raise FileNotFoundError(f"no shard_*.npy files in {path}")
        return np.concatenate([np.load(s) for s in shards])
    return np.load(path)


def remove_overlap(games: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """Drop every row of games that also appears as a row of exclude."""
    seen = {row.tobytes() for row in exclude}
    keep = np.array([row.tobytes() not in seen for row in games], dtype=bool)
    return games[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, required=True, help="number of games")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, help="drop games that appear in this .npy file or shard directory")
    args = parser.parse_args()

    games = generate(args.n, args.seed)
    if args.exclude:
        before = len(games)
        games = remove_overlap(games, load_games(args.exclude))
        print(f"dropped {before - len(games)} games that also appear in {args.exclude}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, games)
    print(f"saved {len(games)} games to {args.out} ({games.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
