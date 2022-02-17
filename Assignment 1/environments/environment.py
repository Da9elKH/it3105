from utils.state import State
from utils.types import ActionList, Action


class ProblemEnvironment:
    def __init__(self):
        self.replay_states = []

    def __action_space(self) -> list[int]:
        """ Get the available actions for this game"""
        pass

    def input_space(self):
        """ Return the input space for this game """
        pass

    def legal_actions(self) -> list[int]:
        """ Defines the legal actions in current state"""
        pass

    def step(self, action: Action) -> tuple[State, Action, float, State, ActionList, bool]:
        """ Run the next action in the given environment """
        pass

    def state(self) -> State:
        """ Returns a tuple containing the state of the game """
        pass

    def __has_failed(self) -> bool:
        """ Check if problem failed (often outside of limits) """
        pass

    def __has_succeeded(self) -> bool:
        """ Check if the problem has been solved """
        pass

    def __in_terminal_state(self) -> bool:
        """ This is used to correctly bootstrap terminal states """
        pass

    def __has_timed_out(self) -> bool:
        """ This is used if environment times out """
        pass

    def is_finished(self) -> bool:
        """ Check if problem is solved """
        return self.__has_succeeded() or self.__has_failed() or self.__has_timed_out()

    def reinforcement(self) -> float:
        """ What is the reward for this episode """
        pass

    def reset(self) -> tuple[State, list[int]]:
        """ Resets the environment to initial state """
        pass

    def store_training_metadata(self, last_episode, current_episode, current_step, state, reinforcement):
        """ Used for printing after an environment is completed """
        pass

    def replay(self, saps, values):
        """ Runs after all episodes """
        pass
