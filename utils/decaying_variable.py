import numpy as np


class DecayingVariable:
    def __init__(self, start=1.0, end=0.01, episodes=300, decay=None, linear=False):
        self.__start = start
        self.__end = end
        self.__current = np.inf
        self.__linear = linear
        self.__episode = 0
        self.__episodes = episodes

        if self.__linear:
            self.__decay = (start - end)/(episodes-1)
        else:
            self.__decay = decay if decay else (end / start) ** (1 / (episodes - 1))

    def __call__(self):
        self.__episode += 1

        if self.__episode >= self.__episodes:
            self.__current = 0
        elif self.__linear:
            self.__current = max(self.__end, min(self.__start, self.__current - self.__decay))
        else:
            self.__current = max(self.__end, min(self.__start, self.__current * self.__decay))

        return self.__current
