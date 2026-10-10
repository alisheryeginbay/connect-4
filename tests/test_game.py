import pytest

from connect4.game import COLS, FIRST, ROWS, SECOND, Game


def play_all(moves: list[int]) -> Game:
    game = Game()
    for col in moves:
        game.play(col)
    return game


def test_pieces_stack_from_bottom():
    game = play_all([3, 3])
    assert game.board[0][3] == FIRST
    assert game.board[1][3] == SECOND
    assert game.heights[3] == 2


def test_full_column_is_illegal():
    game = play_all([0] * ROWS)
    assert 0 not in game.legal_moves()
    with pytest.raises(ValueError):
        game.play(0)


@pytest.mark.parametrize(
    "moves",
    [
        [0, 0, 1, 1, 2, 2, 3],  # horizontal
        [0, 1, 0, 1, 0, 1, 0],  # vertical
        [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3],  # diagonal /
        [6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3],  # diagonal \
    ],
)
def test_first_player_wins(moves):
    game = play_all(moves)
    assert game.winner == FIRST
    assert game.is_over
    assert game.legal_moves() == []


def test_three_in_a_row_is_not_a_win():
    game = play_all([0, 0, 1, 1, 2, 2])
    assert game.winner is None
    assert not game.is_over


def test_draw_fills_the_board():
    # A full-board game with no four in a row (found by random search).
    moves = [3, 6, 1, 3, 1, 3, 6, 4, 5, 5, 3, 4, 3, 1, 1, 0, 4, 3, 0, 4, 6,
             6, 2, 6, 1, 0, 5, 5, 2, 4, 5, 1, 5, 6, 0, 0, 2, 2, 4, 2, 2, 0]
    game = play_all(moves)
    assert len(game.moves) == ROWS * COLS
    assert game.winner is None
    assert game.is_over
    assert game.legal_moves() == []
