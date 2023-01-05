"""
用户自定义包
"""
from .type_check import *
import numpy as np


class Custom(object):

    def fit(self, s):
        return self

    def transform(self, s):
        raise Exception("need to implement")

    def get_params(self) -> dict:
        return dict()

    def set_params(self, params: dict):
        pass


class MapValues(Custom):
    def __init__(self, map_values: dict = None):
        self.map_values = map_values if map_values is not None else dict()

    def transform(self, s: dataframe_type) -> dataframe_type:
        for col in s.columns:
            if col in self.map_values:
                s[col] = np.round(s[col] / self.map_values[col][0] * self.map_values[col][1], 2)
        return s

    def get_params(self) -> dict:
        return {"map_values": self.map_values}

    def set_params(self, params: dict):
        self.map_values = params["map_values"]
