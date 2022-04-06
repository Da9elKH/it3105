from functools import lru_cache
from typing import TypeVar

import numpy as np
from unionfind import UnionFind

from environment import Environment
from misc import Move

PLAYERS = (1, -1)

THexGame = TypeVar("THexGame", bound="Hex")


class HexGame(Environment):
    def __init__(self, size=5, start_player=PLAYERS[0], state=np.zeros((1,))):
        super().__init__()
        self.size = size
        self._start_player = start_player
        self._uf = self._uf_init()

        # These will be updated in state.setter
        self._is_game_over = None
        self._legal_moves = set([])

        self._shadow_state = self._state_init()
        self.state = state

    # ####################
    #    State handling
    # ####################
    @property
    def state(self):
        return self._shadow_state[1:-1, 1:-1]

    @state.setter
    def state(self, value):
        self._shadow_state[1:-1, 1:-1] = value
        self._on_state_updated()

    @property
    def flat_state(self):
        return [self.current_player, *self.state.flatten()]

    @property
    def rotated_cnn_state(self):
        return self._cnn_state(True)

    @property
    def cnn_state(self):
        return self._cnn_state(False)

    def _cnn_state(self, rotate=False):
        cnn_state = self.state if not rotate else self.state[::-1, ::-1]
        player1 = (cnn_state == PLAYERS[0])
        player2 = (cnn_state == PLAYERS[1])
        empty = (cnn_state == 0)

        if self.current_player == PLAYERS[0]:
            to_play = [np.ones(cnn_state.shape), np.zeros(cnn_state.shape)]
        else:
            to_play = [np.zeros(cnn_state.shape), np.ones(cnn_state.shape)]

        return np.moveaxis(
            np.array([player1, player2, empty, *to_play]), 0, 2)

    def _on_state_updated(self):
        self._uf_state_sync()

        # Check if the last move ended the game
        last_player = self._current_player * (-1)
        self._is_game_over = self._uf[last_player].connected("start", "end")

        # Set possible moves
        self._legal_moves = [tuple(x) for x in np.argwhere(self.state == 0).tolist()]

        # If not then update the player
        if self.is_game_over:
            self.current_player = last_player
        else:
            self.current_player = last_player * (-1)

    def _state_init(self):
        shadow_state = np.zeros((self.size + 2, self.size + 2))
        shadow_state[:, 0] = shadow_state[:, -1] = PLAYERS[1]
        shadow_state[0, :] = shadow_state[-1, :] = PLAYERS[0]
        return shadow_state

    # ####################
    #    Game Playing
    # ####################
    def play(self, move: Move):
        if isinstance(move, list):
            move = tuple(move)

        if move in self.legal_moves and not self.is_game_over:
            # Register the move
            self.state[move] = self.current_player
            self._legal_moves.remove(move)

            # Sync uf neighbors
            self._uf_merge_neighbors(move, self.current_player)

            # Update game status
            self._is_game_over = self.uf.connected("start", "end")

            if not self.is_game_over:
                # Switch player
                self.current_player = self.next_player

            # Broadcast move to agents following this state
            self.broadcast_move(move)
        else:
            raise ValueError(f"Move {move} is not allowed")

    @property
    def legal_moves(self):
        if self.is_game_over:
            return []
        return list(self._legal_moves)

    @property
    def legal_binary_moves(self):
        return np.logical_not(self.state).flatten().tolist()

    def transform_move_to_binary_move_index(self, move: Move) -> int:
        return move[0] * self.size + move[1]

    def transform_binary_move_index_to_move(self, binary_move_index: int) -> Move:
        return np.unravel_index(binary_move_index, shape=self.state.shape)

    @property
    def next_player(self):
        return self.current_player * (-1)

    @property
    def _current_player(self):
        return self._start_player * (-1) ** (np.sum(self.state != 0))

    # ####################
    #   Game over check
    # ####################
    @property
    def is_game_over(self):
        return self._is_game_over

    @property
    def result(self):
        return self.current_player

    @property
    def uf(self):
        return self._uf[self.current_player]

    def _uf_state_sync(self):
        # Sync with current state
        for player in PLAYERS:
            for location in np.argwhere(self.state == player).tolist():
                self._uf_merge_neighbors(location, player)

    @classmethod
    @lru_cache(typed=True)
    def _uf_neighbors(cls, location, size):
        neighbors = lambda r, c: [(r + 1, c - 1), (r, c - 1), (r - 1, c), (r - 1, c + 1), (r, c + 1), (r + 1, c)]
        inside = lambda x: 0 <= x <= size + 1
        return [(i, j) for i, j in neighbors(*location) if inside(i) and inside(j)]

    def _uf_merge_neighbors(self, location, player):
        # Merge all neighbors
        location = (location[0] + 1, location[1] + 1)
        neighbors = self._uf_neighbors(location, self.size)

        for neighbor in neighbors:
            if self._shadow_state[location] == self._shadow_state[neighbor]:
                self._uf[player].union(location, neighbor)

    def _uf_init(self):
        # Initiate union-find for both players
        uf = {
            PLAYERS[0]: UnionFind(["start", "end"]),
            PLAYERS[1]: UnionFind(["start", "end"])
        }

        # Connect player edges
        for i in range(self.size + 2):
            uf[PLAYERS[0]].union("start", (0, i))
            uf[PLAYERS[0]].union("end", (self.size + 1, i))
            uf[PLAYERS[1]].union("start", (i, self.size + 1))
            uf[PLAYERS[1]].union("end", (i, 0))

        return uf

    # ####################
    #        Misc
    # ####################
    def copy(self):
        new = self.__class__(start_player=self._start_player, size=self.size, state=self.state.copy())
        return new

    def reset(self, start_player=None):
        self._start_player = start_player if start_player else self._start_player
        self._uf = self._uf_init()
        self.state = 0
        self.broadcast_reset()
