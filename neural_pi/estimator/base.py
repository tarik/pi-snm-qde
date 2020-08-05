import math
import os
import torch
import uuid
import numpy as np
from collections import OrderedDict
from abc import ABC, abstractmethod


class Estimator(ABC):

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        raise NotImplementedError


class MiniBatches:
    """
    Handles mini-batches.
    """

    def __init__(self, x, y, batch_size, randomness=None):
        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._batch_count = math.ceil(self._x.shape[0] / self._batch_size)
        self._randomness = randomness if randomness is not None else Randomness(None)

    def get_batches(self):
        shuffled_indexes = self._randomness.permutation(self._x.shape[0])
        x_shuffled = self._x[shuffled_indexes]
        y_shuffled = self._y[shuffled_indexes]

        mini_batches = []
        for b in range(0, self._batch_count):
            if b != self._batch_count:
                X_batch = x_shuffled[b * self._batch_size:(b + 1) * self._batch_size]
                y_batch = y_shuffled[b * self._batch_size:(b + 1) * self._batch_size]
            else:  # last batch
                X_batch = x_shuffled[b * self._batch_size:]
                y_batch = y_shuffled[b * self._batch_size:]
            mini_batches.append((X_batch, y_batch))

        return mini_batches

    @property
    def batch_count(self):
        return self._batch_count


class Randomness(np.random.RandomState):

    _SEED_MIN = 1
    _SEED_MAX = int(1e9)

    def random_seed(self, low=_SEED_MIN, high=_SEED_MAX, size=None, dtype='l'):
        return self.randint(low=low, high=high, size=size, dtype=dtype)


def evaluate(y_pred, y_true, metrics=[]):
    kwargs = y_to_kwargs(y_pred, y_true)
    results = OrderedDict()
    for metric_func in metrics:
        name = metric_func.__name__
        results[name] = metric_func(**kwargs)
    return results


def y_to_kwargs(y_pred, y_true=None):
    kwargs = dict(
        y_l=y_pred[:, 0],  # lower bound
        y_u=y_pred[:, 1],  # upper bound
        y_p=y_pred[:, 2],  # point prediction
        y=None  # ground truth
    )
    if y_true is not None:
        kwargs['y'] = y_true[:, 0]
    return kwargs


class EarlyStopping:
    """
    Early-stops the training if validation loss does not improve after a given `patience`.

    A modified code from:
    https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, patience=7, verbose=False, delta=0., min_lr=1e-9, output_dir='./temp', **kwargs):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.epoch = None
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.min_lr = min_lr
        self._checkpoint_path = os.path.join(output_dir, 'checkpoint_%s.pt' % uuid.uuid4().hex)

    def __call__(self, epoch, val_loss, model, lr=None):
        if lr < self.min_lr:
            self.early_stop = True
            if self.verbose:
                print(f'Early stopping due to lr < %f...' % self.min_lr)
        score = -val_loss
        if self.best_score is None:
            self.epoch = epoch
            self.best_score = score
            self._save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.epoch = epoch
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Creating a checkpoint...')
        torch.save(model.state_dict(), self._checkpoint_path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self._checkpoint_path))
        return model

    def clean(self):
        os.remove(self._checkpoint_path)
