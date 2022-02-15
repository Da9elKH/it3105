import math
import random
from typing import Union, Optional
from utils.state import State
from utils.types import Action, ActionList

import numpy as np

from utils.decaying_variable import DecayingVariable


class Actor:
    def __init__(
            self,
            discount_factor: float,
            trace_decay: float,
            learning_rate: Union[float, DecayingVariable],
            epsilon: Union[float, DecayingVariable]
    ):
        self.__saps = {}
        self.__eligibilities = {}
        self.__discount_factor = discount_factor  # Gamma (γ): Discount factor for future states
        self.__trace_decay = trace_decay  # Lambda (λ): Trace decay

        self.__learning_rate = 1.0  # Alpha (α): Learning rate
        self.__epsilon = 1.0  # Epsilon (ε) exploration vs exploitation
        self.__e = epsilon
        self.__lr = learning_rate

        self.__states_visited: set[tuple[str, int]] = set()

    def clear(self):
        self.__eligibilities.clear()
        self.__states_visited.clear()

        # Adjust epsilon for each episode
        self.__learning_rate = self.__lr if isinstance(self.__lr, float) else self.__lr()
        self.__epsilon = self.__e if isinstance(self.__e, float) else self.__e()

    """ ELIGIBILITIES """
    def set_eligibility(self, state: State, action: Action):
        self.__states_visited.add((state.binary_string, action))
        self.__eligibilities[(state.binary_string, action)] = 1

    def __eligibility(self, state: str, action: Action):
        # Return, or create and return, the eligibility of this state-action-pair
        return self.__eligibilities.setdefault((state, action), 0)

    def __adjust_eligibility(self, state: str, action: Action):
        self.__eligibilities[(state, action)] = \
            self.__discount_factor * self.__trace_decay * self.__eligibility(state, action)

    """ POLICY FUNCTIONS """
    def next_action(self, state: State, actions: ActionList) -> Optional[Action]:
        if not actions:
            return None
        elif random.random() < self.__epsilon:
            return random.choice(actions)
        else:
            possible_actions = {}
            random.shuffle(actions)
            for action in actions:
                possible_actions[(state.binary_string, action)] = self.__sap(state.binary_string, action)
            return max(possible_actions, key=possible_actions.get)[1]

    def __sap(self, state: str, action: Action):
        # Return, or create and return, the probability of this state-action-pair
        return self.__saps.setdefault((state, action), 0)

    def __adjust_sap(self, state: str, action: Action, td_error: float):
        self.__saps[(state, action)] = \
            self.__sap(state, action) + \
            self.__learning_rate * td_error * self.__eligibility(state, action)

    """ ADJUSTMENTS """
    def adjust(self, td_error):
        for state, action in self.__states_visited:
            self.__adjust_sap(state, action, td_error)
            self.__adjust_eligibility(state, action)

    def get_saps(self):
        return self.__saps