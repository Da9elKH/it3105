from copy import deepcopy, copy
from typing import List

import numpy as np

from environment import Environment
from misc.types import Move


class Nim(Environment):
    def __init__(self, num_stones, max_choice):
        super().__init__()
        self._num_stones = num_stones
        self.num_stones = num_stones
        self.max_choice = max_choice

    def reset(self) -> None:
        self.num_stones = copy(self._num_stones)

    @property
    def is_game_over(self) -> bool:
        """ Returns whether the game is won by one of the players """
        return self.num_stones == 0

    @property
    def legal_moves(self) -> List[int]:
        return list(range(1, min(self.max_choice, self.num_stones) + 1))

    def legal_moves_binary(self):
        moves = np.zeros(self.max_choice)
        for i in self.legal_moves:
            moves[i - 1] = 1
        return moves

    @property
    def result(self) -> int:
        """ Returns the game result from first player perspective """
        if self.current_player == 1:
            return 1
        else:
            return -1

    @property
    def flat_state(self) -> List[int]:
        """ Returns a state to be used together with neural networks """
        return [self.num_stones]

    def execute(self, move: Move):
        """ Executes the move for the current player on the board """
        if move in self.legal_moves:
            self.num_stones -= move
            if not self.is_game_over:
                self.current_player = 1 if self.current_player == 2 else 2

        return self.copy()

    def copy(self):
        """ Deep copy of the state manager """
        return deepcopy(self)
