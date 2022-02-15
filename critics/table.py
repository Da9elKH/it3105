from critics.critic import Critic
import random
from typing import Union
from utils.decaying_variable import DecayingVariable
from utils.state import State


class TableCritic(Critic):
    def __init__(self, discount_factor: float, trace_decay: float, learning_rate: Union[float, DecayingVariable]):
        self.__values = {}
        self.__eligibilities = {}

        self.__discount_factor = discount_factor  # Gamma (γ): Discount factor for future states
        self.__trace_decay = trace_decay  # Lambda (λ): Trace decay
        self.__learning_rate = 1.0  # Alpha (α): Learning rate
        self.__lr = learning_rate

        self.__states_visited: set[str] = set()

    """ ELIGIBILITIES """
    def clear(self):
        self.__states_visited.clear()
        self.__eligibilities.clear()
        self.__learning_rate = self.__lr if isinstance(self.__lr, float) else self.__lr()

    def set_eligibility(self, state: State):
        self.__states_visited.add(state.binary_string)
        self.__eligibilities[state.binary_string] = 1

    def __eligibility(self, state: str):
        # Return, or create and return, the eligibility
        return self.__eligibilities.setdefault(state, 0)

    def __adjust_eligibility(self, state: str):
        self.__eligibilities[state] = \
            self.__discount_factor * self.__trace_decay * self.__eligibility(state)

    """ CRITIC """
    def td_error(self, reinforcement: float, from_state: State, to_state: State, done: bool) -> float:
        return reinforcement + (self.__discount_factor * self.__value(to_state.binary_string))*(1-int(done)) - self.__value(from_state.binary_string)

    def __value(self, state: str):
        # Return, or create and return, the value of this state
        return self.__values.setdefault(state, random.uniform(0, 0.1))

    def __adjust_value(self, state: str, td_error: float):
        self.__values[state] = \
            self.__value(state) + \
            self.__learning_rate * td_error * self.__eligibility(state)

    """ ADJUSTMENTS """
    def adjust(self, td_error: float):
        for state in self.__states_visited:
            self.__adjust_value(state, td_error)
            self.__adjust_eligibility(state)

    def get_values(self):
        print("CRITIC ELIG", self.__eligibilities)
        return self.__values
