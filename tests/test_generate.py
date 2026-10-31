import numpy as np

from connect4.game import Game
from connect4.generate import MAX_MOVES, PAD, generate, load_games, remove_overlap, shard_name


def test_same_seed_same_games():
    assert np.array_equal(generate(100, seed=1), generate(100, seed=1))
    assert not np.array_equal(generate(100, seed=1), generate(100, seed=2))


def test_every_game_replays_legally_to_the_end():
    games = generate(500, seed=0)
    assert games.shape == (500, MAX_MOVES)
    for row in games:
        moves = row[row != PAD].tolist()
        assert (row[len(moves) :] == PAD).all()  # padding only at the tail
        game = Game()
        for col in moves:
            game.play(col)  # raises on an illegal move
        assert game.is_over


def test_load_games_concatenates_shards_in_seed_order(tmp_path):
    for seed in (1, 0):
        np.save(tmp_path / shard_name(seed), generate(10, seed))
    assert np.array_equal(load_games(tmp_path), np.concatenate([generate(10, 0), generate(10, 1)]))


def test_remove_overlap_drops_only_shared_games():
    a, b = generate(50, seed=0), generate(50, seed=1)
    kept = remove_overlap(np.concatenate([a[:5], b]), exclude=a)
    assert len(kept) == 50 and np.array_equal(kept, b)
