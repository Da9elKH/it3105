"""
    INTERFACE FOR CRITIC TYPES
"""
from utils.state import State


class Critic:
    def __init__(self):
        pass

    # TODO: Should be renamed 'delta'
    def td_error(self, reinforcement: float, from_state: State, to_state: State, terminal: bool) -> float:
        pass

    def adjust(self, td_error: float):
        pass

    def set_eligibility(self, state: State):
        pass

    def clear(self):
        pass

    def learn(self):
        pass

    # TODO: SHOULD BE REMOVED?
    def get_values(self):
        pass
