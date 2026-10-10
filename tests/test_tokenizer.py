import numpy as np

from connect4.generate import PAD as MOVE_PAD
from connect4.generate import generate
from connect4.tokenizer import BOS, END, PAD, SEQ_LEN, VOCAB_SIZE, decode, encode, to_string


def test_layout_with_end():
    games = np.full((1, 42), MOVE_PAD, dtype=np.int8)
    games[0, :3] = [3, 3, 4]
    tokens = encode(games, end_token=True)[0]
    assert tokens.shape == (SEQ_LEN,)
    assert tokens[:6].tolist() == [BOS, 3, 3, 4, END, PAD]
    assert (tokens[5:] == PAD).all()
    assert to_string(tokens[:6]) == "BOS 3 3 4 END PAD"


def test_no_end_token_when_disabled():
    tokens = encode(generate(200, seed=0), end_token=False)
    assert not (tokens == END).any()


def test_full_board_game_still_fits_with_end():
    games = np.zeros((1, 42), dtype=np.int8)  # contents don't matter, only length
    tokens = encode(games, end_token=True)[0]
    assert tokens[0] == BOS and tokens[-1] == END


def test_round_trip():
    games = generate(200, seed=0)
    for end_token in (True, False):
        tokens = encode(games, end_token)
        assert tokens.min() >= 0 and tokens.max() < VOCAB_SIZE
        for game, seq in zip(games, tokens):
            assert decode(seq) == game[game != MOVE_PAD].tolist()
