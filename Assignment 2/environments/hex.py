from collections import deque
from functools import lru_cache
from typing import Union, Tuple

import arcade
import arcade.gui
import numpy as np
import networkx as nx
import math
from agents.agent import Agent
from unionfind import UnionFind
from misc.state_manager import StateManager
from copy import deepcopy
from typing import TypeVar, Generic, Tuple, List
from itertools import product
from misc import Move


PLAYERS = (1, -1)

THexGame = TypeVar("THexGame", bound="HexGame")


class HexGame(StateManager):
    def __init__(self, size=5, start_player=PLAYERS[0], state=np.zeros((1,), dtype=np.int8)):
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
    def cnn_state(self):
        player_state = (self.state == self.current_player).astype(np.int8)
        opponent_state = (self.state == self.next_player).astype(np.int8)

        # Transposing the state for player two to be seen as win from top to bottom
        if self.current_player == PLAYERS[1]:
            player_state = player_state.T
            opponent_state = opponent_state.T

        return np.moveaxis(np.array([player_state, opponent_state]), 0, 2)

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
        shadow_state = np.zeros((self.size + 2, self.size + 2), dtype=np.int8)
        shadow_state[:, 0] = shadow_state[:, -1] = PLAYERS[1]
        shadow_state[0, :] = shadow_state[-1, :] = PLAYERS[0]
        return shadow_state

    # ####################
    #    Game Playing
    # ####################
    def play(self, move):
        if isinstance(move, list):
            move = tuple(move)

        if move in self.legal_moves:
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
        return np.logical_not(self.state).astype(np.int8).flatten().tolist()

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


class OldHexGame(StateManager, Generic[THexGame]):
    def __init__(self, size=5, start_player=1):
        super().__init__()
        self.size = size
        self.start_player = start_player
        self.players = (1, 2)
        self.current_player, self.state, self.shadow_state, self._union_find, self._game_over = self.init_values()

    def reset(self):
        self.current_player, self.state, self.shadow_state, self._union_find, self._game_over = self.init_values()

    def init_values(self):
        state = np.zeros((self.size, self.size), dtype=np.int32)
        shadow_state = np.zeros((self.size + 2, self.size + 2), dtype=np.int32)
        shadow_state[:, 0] = shadow_state[:, -1] = self.players[0]
        shadow_state[0, :] = shadow_state[-1, :] = self.players[1]
        union_find = {
            self.players[0]: UnionFind(
                ["start", "end", *[(i, j) for i in range(0, self.size + 2) for j in range(0, self.size + 2)]]),
            self.players[1]: UnionFind(
                ["start", "end", *[(i, j) for i in range(0, self.size + 2) for j in range(0, self.size + 2)]])
        }
        for i in range(self.size + 2):
            union_find[self.players[0]].union("start", (i, self.size + 1))
            union_find[self.players[0]].union("end", (i, 0))
            union_find[self.players[1]].union("start", (0, i))
            union_find[self.players[1]].union("end", (self.size + 1, i))

        return self.start_player, state, shadow_state, union_find, False

    def sync_state(self, state):
        if state.shape == self.state.shape:
            self.state += (self.state - self.state)

    @property
    def union_find(self) -> UnionFind:
        return self._union_find[self.current_player]

    @property
    def is_game_over(self) -> bool:
        return self._game_over

    def update_game_status(self):
        self._game_over = self.union_find.connected("start", "end")

    """ MOVES """
    @property
    def legal_moves(self) -> List[Move]:
        if self.is_game_over:
            return []
        return [tuple(x) for x in np.argwhere(self.state == 0).tolist()]

    @property
    def legal_binary_moves(self):
        return np.logical_not(self.state).astype(np.int32).flatten().tolist()

    def transform_move_to_binary_move_index(self, move: Move) -> int:
        return move[0] * self.size + move[1]

    def transform_binary_move_index_to_move(self, binary_move_index: int) -> Move:
        return np.unravel_index(binary_move_index, shape=self.state.shape)

    @property
    def result(self):
        if self.current_player == 1:
            return 1
        else:
            return -1

    @property
    def flat_state(self):
        bits = lambda s: format(s, f"0{2}b")
        state = [self.current_player, *self.state.flatten()]
        return np.array([float(s) for s in list(''.join([bits(s) for s in state]))], dtype=np.int32)

    @property
    def cnn_state(self):
        player_state = (self.state == self.current_player).astype(np.int32)
        opponent_state = (self.state == self.next_player).astype(np.int32)

        # Transposing the state for player two to be seen as win from top to bottom
        if self.current_player == self.players[1]:
            player_state = player_state.T
            opponent_state = opponent_state.T

        return np.moveaxis(np.array([player_state, opponent_state]), 0, 2)

    @property
    def state_test(self):
        return [self.current_player, *self.state.flatten()]

    def copy(self) -> THexGame:
        return deepcopy(self)

    @property
    def next_player(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    def switch_player(self):
        self.current_player = self.next_player

    def play(self, move: Move):
        self.execute(move)

    def execute(self, move: Move):
        if move in self.legal_moves:
            shadow_move = (move[0] + 1, move[1] + 1)
            self.state[move] = self.current_player
            self.shadow_state[shadow_move] = self.current_player
            self._union_neighbors(shadow_move)

            # Cache game over
            self.update_game_status()

            if not self.is_game_over:
                self.switch_player()

    def _union_neighbors(self, move: Move):
        r, c = move
        neighbors = [(r+1, c-1), (r, c-1), (r-1, c), (r-1, c+1), (r, c+1), (r+1, c)]

        for neighbor in neighbors:
            if neighbor[0] < 0 or neighbor[1] < 0:
                continue
            if self.shadow_state[neighbor] == self.shadow_state[move]:
                self.union_find.union(neighbor, move)