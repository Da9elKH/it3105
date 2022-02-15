import tensorflow as tf
import timeit
from critics.critic import Critic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from utils.state import State


class Network:
    def __init__(self):
        pass


class NeuralNetworkCritic(Critic):
    def __init__(self, discount_factor, learning_rate, trace_decay=None, hidden_size=(15, 20, 10, 5), input_size=4):
        self.__discount_factor = discount_factor  # Gamma (γ): Discount factor for future states
        self.__learning_rate = learning_rate  # Alpha (α): Learning rate

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.__model = self.build_network()
        self.__from_states = []
        self.__targets = []

    def build_network(self):
        model = Sequential()
        model.add(Dense(self.input_size, activation="relu", input_dim=self.input_size))
        for v in self.hidden_size:
            model.add(Dense(v, activation="relu"))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), run_eagerly=True)
        return model

    def get_model(self):
        return self.__model

    def td_error(self, reinforcement, from_state: State, to_state: State, done: bool):

        # Alltid kunne gi ut delta uavhengig av batch
        # Stacke opp state, reinforcement for batch senere

        from_state = tf.convert_to_tensor([from_state.binary_array], dtype=tf.float32)
        to_state = tf.convert_to_tensor([to_state.binary_array], dtype=tf.float32)
        reinforcement = tf.convert_to_tensor(reinforcement, dtype=tf.float32)

        to_state = self.__model(to_state)
        target = reinforcement + self.__discount_factor * to_state * (1 - int(done))
        td_error = target - self.__model(from_state)

        self.__from_states.append(from_state)
        self.__targets.append(target)

        return td_error

    def adjust(self, td_error):
        pass

    def set_eligibility(self, state):
        pass

    def clear(self):
        self.__from_states = []
        self.__targets = []

    def learn(self):
        history = self.__model.fit(tf.concat(self.__from_states, axis=0), tf.concat(self.__targets, axis=0), epochs=min(10, len(self.__from_states)), verbose=0)
        print(history.history['loss'][-1])