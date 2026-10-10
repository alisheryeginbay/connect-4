import numpy as np

from connect4.game import Game
from connect4.generate import MAX_MOVES, PAD, generate


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
