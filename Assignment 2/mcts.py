from misc.state_manager import StateManager
from misc.types import Move
from collections import defaultdict
from typing import TypeVar, Generic, Optional
import random
import numpy as np
import graphviz
from copy import copy
import time

TNode = TypeVar("TNode", bound="Node")


class Node(Generic[TNode]):
    def __init__(self, state: StateManager, actor, parent: Optional[TNode] = None, parent_action: Optional[Move] = None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []

        self.player = self.state.current_player

        self._num_visits = 0
        self._results = defaultdict(int)
        self._untried_actions = self.untried_actions

        self.actor = actor

    @property
    def untried_actions(self):
        legal_moves = copy(self.state.legal_moves)
        random.shuffle(legal_moves)
        self._untried_actions = legal_moves
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    @property
    def n(self):
        return self._num_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.copy()
        next_state.execute(action)
        child_node = Node(next_state, parent=self, parent_action=action, actor=self.actor)
        self.children.append(child_node)
        return child_node

    @property
    def is_terminal_node(self):
        return self.state.is_game_over

    def rollout(self):
        current_rollout_state = self.state.copy()

        while not current_rollout_state.is_game_over:
            possible_moves = current_rollout_state.legal_moves
            action = self._rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.execute(action)

        return current_rollout_state.result

    def back_propagate(self, result):
        self._num_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.back_propagate(result)

    @property
    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c=1.0):
        if self.player == 1:
            choices = [(child.q/child.n) + c * np.sqrt(np.log(self.n)/(1 + child.n)) for child in self.children]
            return self.children[np.argmax(choices)]
        else:
            choices = [(child.q/child.n) - c * np.sqrt(np.log(self.n)/(1 + child.n)) for child in self.children]
            return self.children[np.argmin(choices)]

    def best_action(self, n_simulations=500, time_limit=2.0, use_time=False):
        if use_time:
            start_time = time.time()
            while (time.time() - start_time) < time_limit:
                v = self._tree_policy()
                reward = v.rollout()
                v.back_propagate(reward)
        else:
            for i in range(n_simulations):
                v = self._tree_policy()
                reward = v.rollout()
                v.back_propagate(reward)

        return self.best_child(c=1.0)

    def draw(self):
        g = graphviz.Digraph('G', filename='MCTS.gv')
        node_count = 0
        queue = [self]
        while queue:
            parent = queue.pop(0)
            if parent.children:
                for child in parent.children:
                    child_info = "%d, %d, %d, %d" % (child.state.flat_state[0], child.n, child.q, child.player)
                    parent_info = "%d, %d, %d, %d" % (parent.state.flat_state[0], parent.n, parent.q, parent.player)
                    g.edge(parent_info, child_info, label=str(child.parent_action), id=str(node_count))
                    node_count += 1
                    queue.append(child)
        g.view()

    """ POLICIES """

    @staticmethod
    def _rollout_policy(possible_moves: list[Move]):
        # RANDOM POLICY
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node:
            if not current_node.is_fully_expanded:
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
