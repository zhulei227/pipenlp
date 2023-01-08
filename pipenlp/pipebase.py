from .base import *
import copy
from tqdm import tqdm
import pickle


class Pipe(PipeObject):
    def __init__(self):
        PipeObject.__init__(self)
        self.models = []

    def pipe(self, model):
        self.models.append(model)
        return self

    def fit(self, x, show_process=False):
        x_ = copy.copy(x)
        if show_process:
            for model in tqdm(self.models):
                print(model.name)
                x_ = model.fit(x_).transform(x_)
        else:
            for model in self.models:
                x_ = model.fit(x_).transform(x_)
        return self

    def transform(self, x, show_process=False):
        x_ = copy.copy(x)
        if show_process:
            for model in tqdm(self.models):
                print(model.name)
                x_ = model.transform(x_)
        else:
            for model in self.models:
                x_ = model.transform(x_)
        return x_

    def transform_single(self, x, show_process=False):
        x_ = copy.copy(x)
        if show_process:
            for model in tqdm(self.models):
                print(model.name)
                x_ = model.transform_single(x_)
        else:
            for model in self.models:
                x_ = model.transform_single(x_)
        return x_

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump([model.get_params() for model in self.models], f)

    def load(self, path):
        with open(path, "rb") as f:
            for i, params in enumerate(pickle.load(f)):
                self.models[i].set_params(params)

    def get_params(self) -> dict:
        params = [model.get_params() for model in self.models]
        return {"params": params}

    def set_params(self, params: dict):
        params = params["params"]
        for i, param in enumerate(params):
            self.models[i].set_params(param)

    def auto_check_transform(self, x):
        x_ = copy.copy(x)
        for model in self.models:
            x_ = model.auto_check_transform(x_)
