import tensorflow as tf
from critics.critic import Critic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils.state import State


class NeuralNetworkCritic(Critic):
    def __init__(self, discount_factor, learning_rate, trace_decay=None, hidden_size=(15, 20, 10, 5), input_size=4):
        super().__init__()

        self.__discount_factor = discount_factor  # Gamma (γ): Discount factor for future states
        self.__learning_rate = learning_rate  # Alpha (α): Learning rate

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.__model = self.build_network()
        self.__from_states = []
        self.__targets = []

    def build_network(self):
        """ Initialize the NN with the given depth and width for the problem environment """
        model = Sequential()
        model.add(Dense(self.input_size, activation="relu", input_dim=self.input_size))
        for v in self.hidden_size:
            model.add(Dense(v, activation="relu"))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), run_eagerly=True)
        return model

    def td_error(self, reinforcement, from_state: State, to_state: State, terminal: bool):
        """ Calculate the delta for this given from_state, reward and to_state """
        # Convert all values to tensors
        from_state_values = tf.convert_to_tensor([from_state.array], dtype=tf.float32)
        to_state_values = tf.convert_to_tensor([to_state.array], dtype=tf.float32)
        reinforcement = tf.convert_to_tensor(reinforcement, dtype=tf.float32)

        # Calculate target (to be stored for batch learning)
        target = reinforcement + self.__discount_factor * self.__model(to_state_values) * (1 - int(terminal))
        self.__from_states.append(from_state_values.numpy())
        self.__targets.append(target.numpy())

        # Calculate delta
        delta = target - self.__model(from_state_values)

        return delta

    def adjust(self, td_error: float):
        """ Using batch-learning instead, learn(), to speed up the process"""
        # loss = td_error**2
        # Can adjust gradients from loss
        pass

    def clear(self):
        """ Clear the stored states and target before next episode """
        self.__from_states.clear()
        self.__targets.clear()

    def learn(self) -> float:
        """ Update the weights based on a fit on x as from_values and y as target values """
        epochs = min(10, len(self.__from_states))

        # Fix dimensions before training
        x = tf.concat(self.__from_states, axis=0)
        y = tf.concat(self.__targets, axis=0)

        history = self.__model.fit(x, y, epochs=epochs, verbose=0)
        return history.history['loss'][-1]
