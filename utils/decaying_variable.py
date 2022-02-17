import numpy as np


class DecayingVariable:
    def __init__(self, start=1.0, end=0.01, episodes=300, decay=None, linear=False, episodes_end_value=None):
        self.__start = start
        self.__end = end
        self.__current = np.inf
        self.__linear = linear
        self.__episode = 0
        self.__episodes = episodes
        self.__episodes_end_value = episodes_end_value

        if self.__linear:
            self.__decay = (start - end)/episodes
        else:
            self.__decay = decay if decay else (end / start) ** (1 / episodes)

    def __call__(self):
        self.__episode += 1

        if self.__episode >= self.__episodes:
            # Returns the given end_value, or the lowest decayed value at the end of the episodes
            self.__current = self.__episodes_end_value if self.__episodes_end_value is not None else self.__end
        elif self.__linear:
            # Linear decay based on equal steps
            self.__current = max(self.__end, min(self.__start, self.__current - self.__decay))
        else:
            # Quadratic decay / Epsilon decay
            self.__current = max(self.__end, min(self.__start, self.__current * self.__decay))

        return self.__current
