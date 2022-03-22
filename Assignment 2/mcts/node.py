from typing import TypeVar, Generic, Optional
from misc.types import Move
import numpy as np

TNode = TypeVar("TNode", bound="Node")


class Node(Generic[TNode]):
    def __init__(self, parent: Optional[TNode] = None, move: Optional[Move] = None, player: int = 1):
        self.parent = parent
        self.move = move
        self.children = []
        self.player = player

        self.N = 0
        self.Q = 0

    def value(self, c=1.0):
        if self.N == 0:
            # TODO: 0 or inf?
            return 0
        return (self.Q / self.N) + c * np.sqrt(np.log(self.parent.N) / (self.N + 1))

    def visit(self):
        self.N += 1

    def update_value(self, winner: int):
        self.Q += -1 if winner == self.player else 1

    def distribution(self):
        return dict([(child.move, child.N / self.N) for child in self.children])
