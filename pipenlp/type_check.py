import pandas
import warnings

warnings.filterwarnings("ignore")
series_type = pandas.core.series.Series
dataframe_type = pandas.core.frame.DataFrame


def check_series_type(func):
    def wrapper(*args, **kwargs):
        assert type(args[1]) == series_type
        return func(*args, **kwargs)

    return wrapper


def check_dataframe_type(func):
    def wrapper(*args, **kwargs):
        assert type(args[1]) == dataframe_type
        return func(*args, **kwargs)

    return wrapper
