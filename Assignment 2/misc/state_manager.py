from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from .types import Move

TStateManager = TypeVar("TStateManager", bound="StateManager")


class StateManager(ABC, Generic[TStateManager]):
    def __init__(self):
        self.current_player = 1

    def reset(self) -> None:
        pass

    @property
    @abstractmethod
    def is_game_over(self) -> bool:
        """ Returns whether the game is won by one of the players """
        pass

    @property
    @abstractmethod
    def legal_moves(self) -> list[Move]:
        """ Returns a list of available moves as indices in the state """
        pass

    @property
    @abstractmethod
    def legal_binary_moves(self) -> list[int]:
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
    def flat_state(self) -> list[int]:
        """ Returns a state to be used together with neural networks """
        pass

    @abstractmethod
    def execute(self, move: Move) -> None:
        """ Executes the move for the current player on the board """
        pass

    @property
    @abstractmethod
    def next_player(self) -> int:
        """ Returns the next player to play """
        pass

    @abstractmethod
    def copy(self) -> TStateManager:
        """ Deep copy of the state manager """
        pass