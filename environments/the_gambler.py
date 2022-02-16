from environments.environment import ProblemEnvironment
from itertools import permutations
import arcade
import random
import matplotlib.pylab as plt
from operator import itemgetter
from utils.state import State, StateConstructor
from utils.types import Action, ActionList


class TheGambler(ProblemEnvironment):
    def __init__(self, win_probability, state_space=100):
        self.state_space = state_space
        self.money = random.randint(1, state_space-1)
        self.win_probability = win_probability

        self.__state_constructor = StateConstructor(categorical_state_shape=(state_space,))

    def reset(self) -> tuple[State, ActionList]:
        self.money = random.randint(1, self.state_space)
        return self.state(), self.legal_actions()

    def input_space(self):
        return len(self.__state_constructor((self.state_space,)).array)

    def action_space(self) -> ActionList:
        """ Get the available actions for this game"""
        return list(range(1, self.state_space + 1))

    def legal_actions(self) -> ActionList:
        """ Defines the legal actions in current state"""
        return list(range(1, min(self.money, self.state_space-self.money) + 1))

    def reinforcement(self) -> float:
        if self.__has_succeeded():
            return 1.0
        else:
            return 0.0

    def step(self, action: Action) -> tuple[State, Action, float, State, ActionList, bool]:
        """ Run the next action in the given environment """
        from_state = self.state()

        if random.random() < self.win_probability:
            self.money += action
        else:
            self.money -= action

        return from_state, action, self.reinforcement(), self.state(), self.legal_actions(), self.is_finished()

    def state(self) -> State:
        """ Returns a tuple containing the state of the game """
        return self.__state_constructor((self.money,))

    def __has_failed(self) -> bool:
        if self.money == 0:
            return True
        return False

    def __has_succeeded(self) -> bool:
        if self.money == self.state_space:
            return True
        return False

    # TODO: To be removed
    def has_succeeded(self) -> bool:
        return self.__has_succeeded()

    def is_finished(self) -> bool:
        return self.__has_succeeded() or self.__has_failed()

    def replay(self, saps, values):
        best_policy = dict([(i, []) for i in range(1, self.state_space)])
        for k, v in saps.items():
            best_policy[int(k[0], 2)].append((k[1], v))

        print(best_policy)

        for k, v in best_policy.items():
            best_policy[k] = max(v, key=itemgetter(1))[0]

        lists = sorted(best_policy.items())
        x, y = zip(*lists)

        print(x, y)

        plt.plot(x, y)
        plt.show()
