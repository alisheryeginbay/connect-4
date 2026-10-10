"""Connect 4 rules: a 6x7 board, pieces drop to the lowest empty cell of a column."""

ROWS = 6
COLS = 7

# Cell values. Players are +1 / -1 so that "whose turn" flips with a sign change,
# and a board seen from the current player's side is just `board * player`.
EMPTY = 0
FIRST = 1
SECOND = -1

# Directions to look for four in a row: horizontal, vertical, and both diagonals.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Game:
    """A single Connect 4 game.

    The board is indexed as board[row][col] with row 0 at the BOTTOM,
    so a piece dropped into column c lands at row heights[c].
    """

    def __init__(self) -> None:
        self.board = [[EMPTY] * COLS for _ in range(ROWS)]
        self.heights = [0] * COLS  # number of pieces in each column
        self.moves: list[int] = []  # columns played, in order
        self.player = FIRST  # whose turn it is
        self.winner: int | None = None

    @property
    def is_over(self) -> bool:
        return self.winner is not None or len(self.moves) == ROWS * COLS

    def legal_moves(self) -> list[int]:
        if self.is_over:
            return []
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def play(self, col: int) -> None:
        if col not in self.legal_moves():
            raise ValueError(f"illegal move: column {col}")
        row = self.heights[col]
        self.board[row][col] = self.player
        self.heights[col] += 1
        self.moves.append(col)
        if self._wins_at(row, col):
            self.winner = self.player
        self.player = -self.player

    def _wins_at(self, row: int, col: int) -> bool:
        # Only the piece just placed can create a new four, so we only check lines through it.
        piece = self.board[row][col]
        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == piece:
                    count += 1
                    r, c = r + sign * dr, c + sign * dc
            if count >= 4:
                return True
        return False

    def __str__(self) -> str:
        symbols = {EMPTY: ".", FIRST: "X", SECOND: "O"}
        rows = [" ".join(symbols[v] for v in self.board[r]) for r in reversed(range(ROWS))]
        return "\n".join(rows + [" ".join(str(c) for c in range(COLS))])
