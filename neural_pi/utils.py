import os
import torch
import pandas as pd
import numpy as np
import datetime
from tabulate import tabulate


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def rename(name):
    """
    Sets a new `__name__`.
    """
    def decorator(f):
        f.__name__ = name
        return f
    return decorator


def print_df(df, table_format='fancy_grid'):
    print(tabulate(df, headers='keys', tablefmt=table_format))


def dict_to_dataframe(obj):
    return pd.DataFrame(obj).T


def save_predictions(y_preds, y_test, r_shuffle, r_seed, output_dir):
    """
    `pd.read_csv(file_path, header=[0, 1], index_col=0)`
    """
    r_preds_dfs = []
    for rp in y_preds:
        r_preds_dfs.append(pd.DataFrame(data=rp, columns=['y_l', 'y_u', 'y_p']))
    r_preds_dfs.append(pd.DataFrame(data=y_test, columns=['y_t']))
    y_preds = pd.concat(r_preds_dfs, axis=1, keys=range(len(r_preds_dfs)))
    output_path = os.path.join(output_dir, '%d_%d_predictions.csv' % (r_shuffle, r_seed))
    y_preds.to_csv(output_path)


def torch_to_numpy(dict_):
    for key, value in dict_.items():
        if isinstance(value, torch.Tensor):
            dict_[key] = value.cpu().numpy()
    return dict_


class Timer:

    def __init__(self):
        self._start_time = None
        self._stop_time = None

    def start(self):
        self._start_time = self._now()
        self._stop_time = None
        return self

    def stop(self):
        self._stop_time = self._now()
        return self

    def reset(self):
        self._start_time = None
        self._stop_time = None
        return self

    def duration(self, decimals=1):
        stop_time = self._stop_time or self._now()
        return np.round((stop_time - self._start_time).total_seconds(), decimals)

    @staticmethod
    def _now():
        return datetime.datetime.now()

    @property
    def start_time(self):
        return self._start_time

    @property
    def stop_time(self):
        return self._stop_time


def print_time(string_in):
    print(string_in, '\t -- ', datetime.datetime.now().strftime('%H:%M:%S'))
