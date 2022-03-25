from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Input, add
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from os import path
import numpy as np
import tensorflow as tf
import math


MODELS_FOLDER = "/Users/daniel/Documents/AIProg/Assignments/Assignment 2/models/"


# https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/b04e80409a26896ae0e5f1d4cbca603f9ae4eff2/loss.py
def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true
    zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)
    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)
    return loss


class CNN:
    def __init__(self, model: Model):
        self.model = model

    def predict(self, x):
        return self.model(x)

    def train_on_batch(self, states, distributions, results):
        x, y = np.array(states), {"value": np.array(results), "policy": np.array(distributions)}
        return self.model.train_on_batch(x, y)

    @classmethod
    def from_file(cls, filename):
        """ Returns a network instance from file """
        model = load_model(MODELS_FOLDER + filename, custom_objects={"softmax_cross_entropy_with_logits": softmax_cross_entropy_with_logits})
        return cls(model=model)

    @classmethod
    def build(cls, input_size: int, output_size: int, hidden_size: tuple[int, ...], learning_rate: float, momentum: float):
        """ Initialize the NN with the given depth and width for the problem environment """

        def conv(x):
            x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False, activation="linear", padding='same', data_format="channels_last")(x)
            x = BatchNormalization(axis=1)(x)
            x = swish(x)
            return x

        def residual(input_data):
            x = conv(input_data)
            x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False, activation='linear', padding='same', data_format="channels_last")(x)
            x = BatchNormalization(axis=1)(x)
            x = add([input_data, x])
            x = swish(x)
            return x

        def policy(x):
            x = Conv2D(filters=2, kernel_size=(1, 1), use_bias=False, activation='linear', padding='same', data_format="channels_last")(x)
            x = BatchNormalization(axis=1)(x)
            x = swish(x)
            x = Flatten()(x)
            x = Dense(output_size, use_bias=False, activation='softmax', name='policy')(x)
            return x

        def value(x):
            x = Conv2D(filters=1, kernel_size=(1,1), use_bias=False, activation='linear', padding='same', data_format="channels_last")(x)
            x = BatchNormalization(axis=1)(x)
            x = swish(x)
            x = Flatten()(x)
            x = Dense(20, use_bias=False, activation='linear')(x)
            x = swish(x)
            x = Dense(1, use_bias=False, activation='tanh', name="value")(x)
            return x

        # Model build
        main_input = Input(shape=input_size, name='input')
        x = conv(main_input)

        # Hidden layers
        for i in range(10):
            x = residual(x)

        # Value and policy head
        pol = policy(x)
        val = value(x)

        model = Model(inputs=[main_input], outputs=[pol, val])
        model.compile(
            loss={'value': 'mean_squared_error', 'policy': softmax_cross_entropy_with_logits},
            optimizer=SGD(learning_rate=learning_rate, momentum=momentum),
            loss_weights={'value': 0.5, 'policy': 0.5}
        )

        return cls(model=model)

    """ MISC """
    def save_model(self, suffix):
        num = 1
        name = lambda n: "(%d) " % n + f"{self.__class__.__name__}_" + suffix + ".h5"

        if self.model is not None:
            while path.exists(MODELS_FOLDER + name(num)):
                num += 1
            self.model.save(MODELS_FOLDER + name(num))
        return name(num)

