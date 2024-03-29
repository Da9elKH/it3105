class State:
    def __init__(self, categorical_state, categorical_state_shape, original_state=None, binary_array=True):
        self.__state_shape = categorical_state_shape
        self.__binary_state_shape = tuple([len(format(length, 'b')) for length in categorical_state_shape])

        # Different state storages
        self.__categorical_state = categorical_state
        self.__original_state = original_state
        self.__use_binary_array = binary_array

        assert len(categorical_state) == len(categorical_state_shape), "State shape and state must have same length"
        assert all([categorical_state[i] <= categorical_state_shape[i] for i in range(len(categorical_state_shape))]), "State is outside state shape"

    @property
    def tuple(self) -> tuple[int]:
        """ Function for returning a tuple representation of the state """
        return self.__original_state

    @property
    def binary_string(self) -> str:
        """ Function for returning a binary string representation of the state """
        return ''.join([format(s, f"0{self.__binary_state_shape[i]}b") for i, s in enumerate(self.__categorical_state)])

    @property
    def array(self) -> list[float]:
        """ Function for returning an array representation of the state, used for NN """
        if self.__use_binary_array:
            return [float(s) for s in list(self.binary_string)]
        else:
            return list(self.__original_state)


class StateConstructor:
    def __init__(self, categorical_state_shape: tuple, binary_array=True):
        self.__state_shape = categorical_state_shape
        self.__binary_array = binary_array

    def __call__(self, categorical_state: tuple, original_state: tuple = None) -> State:
        return State(categorical_state, original_state=original_state, categorical_state_shape=self.__state_shape, binary_array=self.__binary_array)
