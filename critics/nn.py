import tensorflow as tf
import timeit
from critics.critic import Critic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from utils.state import State


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

    def td_error(self, reinforcement, from_state: State, to_state: State, terminal: bool):
        from_state_values = tf.convert_to_tensor([from_state.array], dtype=tf.float32)
        to_state_values = tf.convert_to_tensor([to_state.array], dtype=tf.float32)
        reinforcement = tf.convert_to_tensor(reinforcement, dtype=tf.float32)

        to_state_predictions = self.__model(to_state_values)
        target = reinforcement + self.__discount_factor * to_state_predictions * (1 - int(terminal))

        # Store states and targets for batch learning
        self.__from_states.append(from_state_values)
        self.__targets.append(target)

        # Calculate delta
        delta = target - self.__model(from_state_values)

        return delta

    def clear(self):
        self.__from_states.clear()
        self.__targets.clear()

    def learn(self) -> float:
        epochs = min(10, len(self.__from_states))

        # Fix dimensions
        x = tf.concat(self.__from_states, axis=0)
        y = tf.concat(self.__targets, axis=0)

        # Get the latest loss
        history = self.__model.fit(x, y, epochs=epochs, verbose=0)
        return history.history['loss'][-1]
