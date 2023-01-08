import copy

import numpy as np
import pandas
import datetime
import warnings

warnings.filterwarnings("ignore")
series_type = pandas.core.series.Series
dataframe_type = pandas.core.frame.DataFrame
dict_type = dict


def check_series_type(func):
    def wrapper(*args, **kwargs):
        assert type(args[1]) == series_type
        return func(*args, **kwargs)

    return wrapper


def check_dict_type(func):
    def wrapper(*args, **kwargs):
        assert type(args[1]) == dict_type
        return func(*args, **kwargs)

    return wrapper


def check_dataframe_type(func):
    def wrapper(*args, **kwargs):
        assert type(args[1]) == dataframe_type
        return func(*args, **kwargs)

    return wrapper


class PipeObject(object):
    """
    name:模块名称，如果为空默认为self.__class__
    input_col_names:输入数据的columns
    output_col_names:输出数据的columns
    transform_check_max_number_error:在auto_check_transform时允许的最大数值误差
    skip_check_transform_type:在auto_check_transform时是否跳过类型检测(针对一些特殊数据类型，比如稀疏矩阵)
    """

    def __init__(self, name=None, transform_check_max_number_error=1e-5, skip_check_transform_type=False):
        if name is None:
            name = self.__class__
        self.name = name
        self.input_col_names = None
        self.output_col_names = None
        self.transform_check_max_number_error = transform_check_max_number_error
        self.skip_check_transform_type = skip_check_transform_type

    def fit(self, s):
        return self

    def transform(self, s):
        """
        批量接口
        """
        raise Exception("need to implement")

    def transform_single(self, s: dict_type):
        """
        当条数据接口，生产用
        """
        raise Exception("need to implement")

    def auto_check_transform(self, s):
        """
        自动测试批量接口和单条数据接口
        """
        # 预测数据
        batch_transform = self.transform(s)  # 注意:transform可能会修改s
        single_transform = []
        single_operate_times = []
        if type(s) == dataframe_type:
            for record in s.to_dict("record"):
                start_time = datetime.datetime.now()
                single_transform.append(self.transform_single(record))
                end_time = datetime.datetime.now()
                single_operate_times.append((end_time - start_time).microseconds / 1000)
            single_transform = pandas.DataFrame(single_transform)
        else:  # series_type
            for value in s.values:
                start_time = datetime.datetime.now()
                single_transform.append(self.transform_single({self.input_col_names[0]: value}))
                end_time = datetime.datetime.now()
                single_operate_times.append((end_time - start_time).microseconds / 1000)
            single_transform = pandas.DataFrame(single_transform)
        # 都转换为dataframe,方便统一检测
        new_batch_transform = batch_transform
        if type(batch_transform) == series_type:
            new_batch_transform = pandas.DataFrame({self.output_col_names[0]: batch_transform.values})
        # 检验
        # 检验1:输出shape是否一致
        if new_batch_transform.shape != single_transform.shape:
            raise Exception(
                "({})  module shape error , batch shape is {} , single  shape is {}".format(
                    self.name, new_batch_transform.shape, single_transform.shape))

        for col in new_batch_transform.columns:
            # 检验2:输出名称是否一致
            if col not in single_transform.columns:
                raise Exception(
                    "({})  module column error,the batch output column {} not in single output".format(
                        self.name, col))
            # 检验3:数据类型是否一致
            if not self.skip_check_transform_type and new_batch_transform[col].dtype != single_transform[col].dtype:
                raise Exception(
                    "({})  module type error,the column {} in batch is {},while in single is {}".format(
                        self.name, col, new_batch_transform[col].dtype, single_transform[col].dtype))
            # 检验4:数值是否一致
            col_type = str(new_batch_transform[col].dtype)
            batch_col_values = new_batch_transform[col].values
            single_col_values = single_transform[col].values
            if "int" in col_type or "float" in col_type:
                try:
                    batch_col_values = batch_col_values.to_dense()  # 转换为dense
                except:
                    pass
                error_index = np.argwhere(
                    np.abs(batch_col_values - single_col_values) > self.transform_check_max_number_error)
                if len(error_index) > 0:
                    raise Exception(
                        "({})  module value error,the top 5 index {} in column {} ,batch is {},single is {},"
                        "current transform_check_max_number_error is {}".format(
                            self.name, error_index[:5], col, batch_col_values[error_index][:5],
                            single_col_values[error_index][:5], self.transform_check_max_number_error))
            else:
                error_index = np.argwhere(batch_col_values != single_col_values)
                if len(error_index) > 0:
                    raise Exception(
                        "({})  module value error,the top 5 index {} in column {} ,batch is {},single is {}".format(
                            self.name, error_index[:5], col, batch_col_values[error_index][:5],
                            single_col_values[error_index][:5]))
        print("({})  module transform check [success], single transform speed:[{}]ms/it".format(self.name, np.round(
            np.mean(single_operate_times), 2)))
        return batch_transform

    def get_params(self) -> dict_type:
        return {"name": self.name,
                "input_col_names": self.input_col_names,
                "output_col_names": self.output_col_names,
                "transform_check_max_number_error": self.transform_check_max_number_error,
                "skip_check_transform_type": self.skip_check_transform_type}

    def set_params(self, params: dict_type):
        self.name = params["name"]
        self.input_col_names = params["input_col_names"]
        self.output_col_names = params["output_col_names"]
        self.transform_check_max_number_error = params["transform_check_max_number_error"]
        self.skip_check_transform_type = params["skip_check_transform_type"]
