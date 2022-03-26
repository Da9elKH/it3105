from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, List
from .types import Move

TStateManager = TypeVar("TStateManager", bound="StateManager")


class StateManager(ABC, Generic[TStateManager]):
    def __init__(self):
        self.current_player = None
        self._reset_hooks: List[Callable] = []
        self._move_hooks: List[Callable] = []

    @property
    @abstractmethod
    def is_game_over(self) -> bool:
        """ Returns whether the game is won by one of the players """
        pass

    @property
    @abstractmethod
    def legal_moves(self) -> List[Move]:
        """ Returns a list of available moves as indices in the state """
        pass

    @property
    @abstractmethod
    def legal_binary_moves(self) -> List[int]:
        """ Returns a list of ones and zeros, where ones are legal moves """
        pass

    @abstractmethod
    def transform_binary_move_index_to_move(self, binary_move_index: int) -> Move:
        """ Returns a move from a binary move index """
        pass

    @abstractmethod
    def transform_move_to_binary_move_index(self, move: Move) -> int:
        """ Returns a binary move index from a move"""
        pass

    @property
    @abstractmethod
    def result(self) -> int:
        """ Returns the game result from first player perspective """
        pass

    @property
    @abstractmethod
    def flat_state(self) -> List[int]:
        """ Returns a state to be used together with neural networks """
        pass

    @abstractmethod
    def play(self, move: Move) -> None:
        """ Executes the move for the current player on the board """
        [hook() for hook in self._move_hooks]

    @property
    @abstractmethod
    def next_player(self) -> int:
        """ Returns the next player to play """
        pass

    @abstractmethod
    def copy(self) -> TStateManager:
        """ Deep copy of the state manager """
        pass

    def reset(self) -> None:
        pass

    def broadcast_reset(self):
        [hook() for hook in self._reset_hooks]

    def broadcast_move(self, move: Move):
        [hook(move) for hook in self._move_hooks]

    def register_reset_hook(self, function: Callable):
        self._reset_hooks.append(function)

    def register_move_hook(self, function: Callable):
        self._move_hooks.append(function)
