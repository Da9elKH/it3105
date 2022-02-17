from environments.environment import ProblemEnvironment
import gym
from utils.state import State, StateConstructor
from utils.types import Action, ActionList
from scipy.stats import binned_statistic


class AIGym(ProblemEnvironment):
    def __init__(self, environment, time_out=300, state_shape=(5, 5)):
        super().__init__()

        self.env: gym.Env = gym.make(environment)
        self.data = None
        self.done = False
        self.rounds = 0
        self.T = time_out
        self.reward = -1

        self.__state_shape = state_shape
        self.__state_constructor = StateConstructor(state_shape, binary_array=False)

    def __action_space(self) -> list[int]:
        """ Get the available actions for this game"""
        return list(range(self.env.action_space.n))

    def input_space(self):
        """ Return the input space for this game """
        return self.env.observation_space.shape[0]

    def legal_actions(self) -> list[int]:
        """ Defines the legal actions in current state"""
        return self.__action_space()

    def step(self, action: Action) -> tuple[State, Action, float, State, ActionList, bool]:
        """ Run the next action in the given environment """
        from_state = self.state()
        state, reward, done, _ = self.env.step(action)

        self.data = state
        self.done = done
        self.reward = reward
        self.rounds += 1

        return from_state, action, self.reinforcement(), self.state(), self.legal_actions(), done

    def state(self) -> State:
        """ Returns a tuple containing the state of the game """
        return self.__state_constructor(self.__bins(), tuple(self.data))

    def __bins(self) -> tuple[int]:
        """ Function for bucketizing the state """
        low = self.env.observation_space.low
        high = self.env.observation_space.high

        bin_state = tuple(
            [
                binned_statistic(
                    max(low[i], min(high[i], self.data[i])),
                    max(low[i], min(high[i], self.data[i])),
                    bins=self.__state_shape[i],
                    range=(low[i], high[i])
                )[2][0] for i in range(len(self.data))
            ]
        )

        return bin_state

    def __has_failed(self) -> bool:
        """ Check if problem failed (often outside of limits) """
        return False

    def __has_succeeded(self) -> bool:
        """ Check if the problem has been solved """
        return self.done

    def __in_terminal_state(self) -> bool:
        """ This is used to correctly bootstrap terminal states """
        return self.done

    def __has_timed_out(self) -> bool:
        """ This is used if environment times out """
        return self.rounds >= self.T

    def is_finished(self) -> bool:
        """ Check if problem is solved """
        return self.__has_succeeded() or self.__has_failed() or self.__has_timed_out()

    def reinforcement(self) -> float:
        """ What is the reward for this episode """
        return self.reward

    def reset(self) -> tuple[State, list[int]]:
        """ Resets the environment to initial state """
        state = self.env.reset()
        self.data = state
        self.rounds = 0
        self.done = False

        return self.state(), self.legal_actions()

    def store_training_metadata(self, last_episode, current_episode, current_step, state, reinforcement):
        """ Used for printing after an environment is completed """
        pass

    def replay(self, saps, values):
        """ Runs after all episodes """
        pass