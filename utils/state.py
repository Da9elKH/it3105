class State:
    def __init__(self, state, state_shape):
        self.__state_shape = state_shape
        self.__binary_state_shape = tuple([len(format(length, 'b')) for length in state_shape])
        self.__state = state

        assert len(state) == len(state_shape), "State shape and state must have same length"
        assert all([state[i] <= state_shape[i] for i in range(len(state_shape))]), "State is outside state shape"

    @property
    def tuple(self) -> tuple[int]:
        return self.__state

    @property
    def binary_string(self) -> str:
        return ''.join([format(s, f"0{self.__binary_state_shape[i]}b") for i, s in enumerate(self.__state)])

    @property
    def binary_array(self) -> list[float]:
        return [float(s) for s in list(self.binary_string)]


class StateConstructor:
    def __init__(self, state_shape: tuple):
        self.__state_shape = state_shape

    def __call__(self, state: tuple):
        return State(state, self.__state_shape)