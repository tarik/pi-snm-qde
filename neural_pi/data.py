import numpy as np
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

from neural_pi.estimator.base import Randomness


class DataLoader(ABC):

    def __init__(self, shuffle=False, standardize=False, crop=None):
        self._shuffle = shuffle
        self._standardize = standardize
        self._crop = crop
        self.standardizer_train = Standardizer()  # based on training set
        self.standardizer = Standardizer()  # based on full dataset

    def load(self, train_size=0.9, val_size=0., test_size=None, seed=None, **kwargs):
        """
        Loads or generates the dataset. Returns training, validation and test set.

        :param seed: int or None
            Random seed for shuffling or generating the dataset.
        :return: tuple
            Tuple of tuples of Numpy arrays `(x_train, y_train), (x_val, y_val), (x_test, y_test)`.
        """
        x, y = self.generate(seed=seed, **kwargs)
        self.standardizer.fit(x=x, y=y)
        if self._crop is not None:
            crop_size = int(round(self._crop * x.shape[0]))
            x = x[:crop_size]
            y = y[:crop_size]
        splits = split_dataset(x, y, sizes=[train_size, val_size, test_size])
        if self._standardize:  # Standardization based on the training set
            self.standardizer_train.fit(splits[0][0], splits[0][1])
            norm_splits = []
            for x_, y_ in splits:
                if x_ is not None and y_ is not None:
                    x_, y_ = self.standardizer_train.transform(x_, y_, copy=False)
                norm_splits.append((x_, y_))
            splits = norm_splits
        return splits

    @abstractmethod
    def generate(self, seed=None, **kwargs):
        """
        :param seed: int or None
            Random seed for shuffling or generating the dataset.
        :return: tuple
            Tuple of Numpy arrays `(x, y)`.
        """
        return NotImplemented

    @property
    def normalizer(self):
        return self.standardizer_train

    def inverse_y(self, y, copy=True):
        if self._standardize:
            is_numpy = type(y) is np.ndarray
            is_torch = type(y) is torch.Tensor
            if copy:
                if is_numpy:
                    y = y.copy()
                elif is_torch:
                    y = y.clone()
                else:
                    raise TypeError('Unsupported type %s.' % type(y))
            if is_numpy:
                y *= self.standardizer_train.std[1]
                y += self.standardizer_train.mean[1]
            else:
                y *= torch.Tensor(self.standardizer_train.std[1]).to(y.device)
                y += torch.Tensor(self.standardizer_train.mean[1]).to(y.device)
        return y

    def standardize_y(self, y, copy=True):
        """
        The target values are standardized with respect to the training set. This function
        is de-standardizing and then again standardizing with respect to the whole dataset.
        This is useful mainly for the evaluation.
        """
        y = self.inverse_y(y, copy)
        is_numpy = type(y) is np.ndarray
        if is_numpy:
            y -= self.standardizer.mean[1]
            y /= self.standardizer.std[1]
        else:
            y -= torch.Tensor(self.standardizer.mean[1]).to(y.device)
            y /= torch.Tensor(self.standardizer.std[1]).to(y.device)
        return y

    @staticmethod
    def shuffle(data, seed=None):
        indices = np.arange(len(data))
        Randomness(seed).shuffle(indices)
        return data[indices]


class FileDataset(DataLoader):

    def __init__(self, file_path, **kwargs):
        super().__init__(**kwargs)
        self._file_path = file_path

    def generate(self, seed=None, **kwargs):
        data = np.loadtxt(self._file_path, skiprows=0, dtype='float32')
        if self._shuffle:
            data = self.shuffle(data, seed)
        x = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        return x, y


class ShuffledDataset(DataLoader):

    def __init__(self, data_path, shuffle_path, **kwargs):
        super().__init__(**kwargs)
        self._data_path = data_path
        self._shuffle_path = shuffle_path

    def generate(self, shuffle=None, **kwargs):
        data = np.loadtxt(self._data_path, skiprows=0, dtype='float32')
        if self._shuffle:
            shuffles = load_shuffles(self._shuffle_path)
            shuffle_id = self._shuffle if type(self._shuffle) is int and shuffle is None else shuffle
            indices = shuffles[shuffle_id]
            data = data[indices]
        x = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        return x, y

    @property
    def shuffle_id(self):
        return self._shuffle

    @shuffle_id.setter
    def shuffle_id(self, value):
        self._shuffle = value


def save_shuffles(input_path, output_path, number=100, seed=None):
    data = np.loadtxt(input_path, skiprows=0, dtype='float32')
    data_count = len(data)
    shuffles = np.zeros((number, data_count), dtype=np.int)
    indices = np.arange(len(data))
    for i in range(number):
        shuffles[i] = indices.copy()
        Randomness(seed).shuffle(shuffles[i])
    if output_path:
        np.save(output_path, shuffles)
    return shuffles


def load_shuffles(file_path):
    return np.load(file_path)


class SyntheticDataset:

    def __init__(self, **kwargs):
        self._data = None

    def load(self, seed=None, **kwargs):
        randomness = Randomness(seed)

        X_START = -6
        X_END = 6
        N_SAMPLES = 2000
        TRN_SIZE = 1000  # 1000
        VAL_SIZE = 250  # 250
        SIGMA_EXP = -0.15
        GAMMA = 0.95
        ALPHA = 1 - GAMMA

        x = np.linspace(X_START, X_END, N_SAMPLES)
        mu = np.sin(x)
        sigma = np.exp(-abs(x)) ** SIGMA_EXP - 1
        q_l = stats.norm.ppf(ALPHA / 2, loc=mu, scale=sigma)
        q_u = stats.norm.ppf(GAMMA + ALPHA / 2, loc=mu, scale=sigma)
        y = randomness.normal(loc=mu, scale=sigma)

        x_train = randomness.normal(
            loc=(X_START + X_END) / 2,
            scale=(X_END - X_START) / 2,
            size=2 * TRN_SIZE
        )
        x_train = x_train[X_START <= x_train]
        x_train = x_train[x_train <= X_END]
        x_train = x_train[:TRN_SIZE]
        y_train = randomness.normal(
            loc=np.sin(x_train),
            scale=np.exp(-abs(x_train)) ** SIGMA_EXP - 1
        )
        x_val = randomness.normal(
            loc=(X_START + X_END) / 2,
            scale=(X_END - X_START) / 2,
            size=2 * VAL_SIZE
        )
        x_val = x_val[X_START <= x_val]
        x_val = x_val[x_val <= X_END]
        x_val = x_val[:VAL_SIZE]
        y_val = randomness.normal(
            loc=np.sin(x_val),
            scale=np.exp(-abs(x_val)) ** SIGMA_EXP - 1
        )

        self._data = dict(
            x=x,
            mu=mu,
            sigma=sigma,
            q_l=q_l,
            q_u=q_u,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val
        )

        x_train = self.deflatten(x_train)
        y_train = self.deflatten(y_train)
        x_val = self.deflatten(x_val)
        y_val = self.deflatten(y_val)
        return x_train, y_train, x_val, y_val

    def deflatten(self, a):
        a = np.reshape(a, (a.shape[0], 1)).astype(np.float32)
        return a

    @property
    def data(self):
        return self._data


# --------------------------------------------------------------------------------------------------

class Standardizer:

    def __init__(self):
        self._scaler_x = StandardScaler(with_mean=True, with_std=True)
        self._scaler_y = StandardScaler(with_mean=True, with_std=True)

    def fit(self, x=None, y=None):
        self._scaler_x.fit(x)
        self._scaler_y.fit(y)
        return self

    def transform(self, x=None, y=None, copy=True):
        x_ = self._scaler_x.transform(x, copy)
        y_ = self._scaler_y.transform(y, copy)
        return self._drop_none(x_, y_)

    def fit_transform(self, x=None, y=None, copy=True):
        return self.fit(x, y).transform(x, y, copy)

    def inverse_transform(self, x=None, y=None, copy=True):
        x_ = self._scaler_x.inverse_transform(x, copy)
        y_ = self._scaler_y.inverse_transform(y, copy)
        return self._drop_none(x_, y_)

    @property
    def mean(self):
        return self._scaler_x.mean_, self._scaler_y.mean_

    @property
    def std(self):
        return np.sqrt(self._scaler_x.var_), np.sqrt(self._scaler_y.var_)

    @property
    def var(self):
        return self._scaler_x.var_, self._scaler_y.var_

    @staticmethod
    def _drop_none(*args):
        return tuple(filter(lambda v: v is not None, args))


# --------------------------------------------------------------------------------------------------

def split_dataset(x, y, sizes, use_none=True):
    """
    Splits `x` and `y` according the given `sizes` (split proportions).

    :param x: Numpy array
        Features.
    :param y: Numpy array
        Targets.
    :param sizes: list
         A list of float or None values representing split proportions. The sum must be less or equal to 1.
    :param use_none:
        If True and some of the sizes is set to zero, tuple `(None, None)` is returned,
        otherwise an empty Numpy array is returned.
    :return: tuple
        Tuple of tuples of Numpy arrays `((x, y), ...)`.
    """
    sizes_ = complement(sizes, 1.)

    if np.sum(sizes_) > 1:
        raise ValueError('The sum of split proportions must be less or equal to 1.')
    if x.shape[0] != y.shape[0]:
        raise ValueError('Non-matching dimensions of `x` and `y`.')

    splits = []
    while len(sizes_) > 0:
        split_size = int(round(sizes_[0] * x.shape[0]))
        x_split = x[:split_size]
        y_split = y[:split_size]
        if use_none and x_split.shape[0] == 0:
            x_split = None
            y_split = None
        splits.append((x_split, y_split))
        x = x[split_size:]
        y = y[split_size:]
        sizes_ = np.divide(sizes_[1:], 1 - np.sum(sizes_[0]))

    return tuple(splits)


def complement(values, maximum):
    """
    Replaces `None` with a complement to make the sum equal to the given `maximum`. If no `None`,
    `values` are not changed.

    :param values: list
        A list or tuple of float or None values. Must be less or equal to `maximum`.
    :param maximum: float
        The expected sum of all values.
    :return: Numpy array
        Numpy array with the sum equal to `maximum` if `None` was present.
    """
    values = list(values)
    sum_ = np.sum(drop_none(values))
    if sum_ > maximum:
        raise ValueError('The sum is greater than the set total.')
    none_indices = [i for i, v in enumerate(values) if v is None]
    if none_indices:
        portion = (maximum - sum_) / len(none_indices)
        for i in none_indices:
            values[i] = portion
    return np.array(values)


def drop_none(values):
    return list(filter(lambda x: x is not None, values))
