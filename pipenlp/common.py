"""
通用包
"""
from .type_check import *
import numpy as np
import scipy.stats as ss


class Common(object):

    def fit(self, s):
        raise Exception("need to implement")

    def transform(self, s):
        raise Exception("need to implement")

    def get_params(self) -> dict:
        return dict()

    def set_params(self, params: dict):
        pass


class Normalization(Common):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    -------------------------
    output type:pandas.dataframe
    | 0 | 1 | 2 | 3 |
    | 45| 56| 45| 45|
    | 55| 34| 56| 23|
    """

    def __init__(self, normal_range=100, normal_type="cdf", std_range=10):
        """
        :param normal_range:
        :param normal_type: cdf,range
        :param std_range:
        """
        Common.__init__(self)
        self.normal_range = normal_range
        self.normal_type = normal_type
        self.std_range = std_range
        self.mean_std = dict()

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        if self.normal_type == "cdf":
            for col in s.columns:
                col_value = s[col]
                mean = np.median(col_value)
                std = np.std(col_value) * self.std_range
                self.mean_std[col] = (mean, std)
            for col in s.columns:
                s[col] = np.round(
                    ss.norm.cdf((s[col] - self.mean_std[col][0]) / self.mean_std[col][1]) * self.normal_range, 2)
        elif self.normal_type == "range":
            for col in s.columns:
                s[col] = self.normal_range * s[col]
        return s

    def transform(self, s: dataframe_type) -> dataframe_type:
        if self.normal_type == "cdf":
            for col in s.columns:
                if col in self.mean_std:
                    s[col] = np.round(
                        ss.norm.cdf((s[col] - self.mean_std[col][0]) / self.mean_std[col][1]) * self.normal_range, 2)
        elif self.normal_type == "range":
            for col in s.columns:
                if col in self.mean_std:
                    s[col] = self.normal_range * s[col]
        return s

    def get_params(self) -> dict:
        return {"mean_std": self.mean_std, "normal_range": self.normal_range, "normal_type": self.normal_type}

    def set_params(self, params: dict):
        self.mean_std = params["mean_std"]
        self.normal_range = params["normal_range"]
        self.normal_type = params["normal_type"]
