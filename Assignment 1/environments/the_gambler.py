import random
import matplotlib.pylab as plt
from operator import itemgetter
from utils.state import State, StateConstructor
from utils.types import Action, ActionList
from environments.environment import ProblemEnvironment


class TheGambler(ProblemEnvironment):
    def __init__(self, win_probability, state_space=100, time_out=300):
        self.state_space = state_space
        self.money = random.randint(1, state_space - 1)
        self.win_probability = win_probability

        self.T = time_out
        self.rounds = 0
        self.__state_constructor = StateConstructor(categorical_state_shape=(state_space,), binary_array=False)
        self.reinforcements = []

    def reset(self) -> tuple[State, ActionList]:
        """ Reset environment before each episode """
        self.money = random.randint(1, self.state_space - 1)
        self.rounds = 0

        return self.state(), self.legal_actions()

    def input_space(self):
        """ The size of the input space used for NN """
        return len(self.__binary_state())

    def legal_actions(self) -> ActionList:
        """ Defines the legal actions in current state"""
        return list(range(1, min(self.money, self.state_space-self.money) + 1))

    def reinforcement(self) -> float:
        """ Only rewards the success state """
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

        self.rounds += 1

        return from_state, action, self.reinforcement(), self.state(), self.legal_actions(), self.__in_terminal_state()

    def __binary_state(self) -> tuple:
        """ Function to create a binary state for NN """
        #return tuple((float(self.money),))
        return tuple([1 if self.money == m else 0 for m in range(0, self.state_space + 1)])

    def state(self) -> State:
        """ Returns a State-object """
        binary_state = self.__binary_state()
        return self.__state_constructor((self.money,), binary_state)

    def __has_failed(self) -> bool:
        if self.money == 0:
            return True
        return False

    def __has_succeeded(self) -> bool:
        if self.money == self.state_space:
            return True
        return False

    def __in_terminal_state(self) -> bool:
        return self.__has_succeeded() or self.__has_failed()

    def __has_timed_out(self) -> bool:
        if self.rounds >= self.T:
            return True
        return False

    def is_finished(self) -> bool:
        return self.__has_succeeded() or self.__has_failed() or self.__has_timed_out()

    def store_training_metadata(self, last_episode, current_episode, current_step, state, reinforcement):
        """ This stores data for each step in each episode """
        self.reinforcements.append(reinforcement)

    def replay(self, saps, values):
        """ This function is called at the end of the training, to replay / show data """

        best_policy = dict([(i, []) for i in range(1, self.state_space)])
        for k, v in saps.items():
            best_policy[int(k[0], 2)].append((k[1], v))

        for k, v in best_policy.items():
            best_policy[k] = max(v, key=itemgetter(1))[0]

        lists = sorted(best_policy.items())
        x, y = zip(*lists)

        plt.stem(x, y)
        plt.show()

        """
        import pandas as pd

        x, y = zip(*enumerate(self.reinforcements))
        
        N = 10
        y_new = pd.Series(y).rolling(window=N).mean().iloc[N - 1:].values
        x_new = pd.Series(x).rolling(window=N).mean().iloc[N - 1:].values

        plt.plot(x_new, y_new)
        plt.show()
        """

