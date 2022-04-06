import math
from typing import TypeVar, Generic, Optional

from misc.types import Move

TNode = TypeVar("TNode", bound="Node")


class Node(Generic[TNode]):
    def __init__(self, parent: Optional[TNode] = None, move: Optional[Move] = None, player: int = 1):
        self.parent = parent
        self.move = move
        self.children = {}
        self.player = player

        self.N = 0
        self.Q = 0

    def value(self, c=1.0):
        score = 0 if self.N == 0 else (self.Q / self.N)
        explr = c * math.sqrt(math.log(self.parent.N) / (self.N + 1))
        return score + explr

    def visit(self):
        self.N += 1

    def update_value(self, winner: int):
        self.Q += -1 if winner == self.player else 1

    def distribution(self):
        return {move: child.N/self.N for move, child in self.children.items()}
