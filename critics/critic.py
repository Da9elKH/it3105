"""
    INTERFACE FOR CRITIC TYPES
"""
from utils.state import State


class Critic:
    def __init__(self):
        pass

    def td_error(self, reinforcement: float, from_state: State, to_state: State, terminal: bool) -> float:
        """ This function should return delta based on the states and rewards """
        pass

    def adjust(self, td_error: float):
        """ This function is called at the end of the episode for adjustments of eligibilities and state-values """
        pass

    def set_eligibility(self, state: State):
        """ This function sets eligibility to one if used """
        pass

    def clear(self):
        """ This functions runs before the episode and is for clearing variables """
        pass

    def learn(self):
        """ This function runs at the end of the episode and is used for batch learning """
        pass

    def get_values(self):
        """ This function is used to get the state evaluations for replay presentation """
        pass
