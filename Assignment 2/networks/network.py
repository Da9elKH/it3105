from tensorflow.keras.models import Model, load_model
from typing import NamedTuple
from os import path
from config import ROOT_DIR
from misc import LiteModel
import numpy as np
MODELS_FOLDER = f"{ROOT_DIR}/models/"


class ModelInfo(NamedTuple):
    name: str
    type: str
    size: int


class Network:
    def __init__(self, model: Model):
        self.model = model

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if isinstance(self.model, LiteModel):
            return self.model.predict_single(x)
        else:
            return self.model(x)

    def train_on_batch(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        return self.model.train_on_batch(x, y, return_dict=True)

    def fit(self, x, y, **params):
        return self.model.fit(x, y, **params)

    def save_model(self, suffix):
        num = 1
        name = lambda n: "(%d) " % n + f"{self.__class__.__name__}_" + suffix + ".h5"

        if self.model is not None:
            while path.exists(MODELS_FOLDER + name(num)):
                num += 1
            self.model.save(MODELS_FOLDER + name(num))
        return name(num)

    @classmethod
    def info_from_path(cls, filename) -> ModelInfo:
        """ Analyzes the filename and returns some information """
        name = lambda x: x.split(".")[0].split("_")[-1][1:]
        type = lambda x: x.split(" ")[1].split("_")[0]
        size = lambda x: x.split(" ")[1].split("_")[1][1:]
        return ModelInfo(name=name(filename), type=type(filename), size=int(size(filename)))

    @classmethod
    def from_file(cls, filename):
        """ Returns a network instance from file """
        model = load_model(MODELS_FOLDER + filename)
        return cls(model=model)

    @classmethod
    def build_from_config(cls, input_shape: tuple, output_size: int, **params):
        pass

    @classmethod
    def build(cls, **params):
        pass
