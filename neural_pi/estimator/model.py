"""
Contains the implementation of model ensembles for quality-driven (QD) and mean-variance (MVE)
methods.

NOTE: Excuse the redundant code... It is a legacy of the research phase, where we did not aim for
generalization and reusability, and we were rather testing new ideas without affecting the existing
working models.
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .functions import picp, mpiw, cross, mse, pimse, within
from .base import Estimator, MiniBatches, evaluate, Randomness, EarlyStopping, y_to_kwargs
from ..utils import torch_to_numpy
from . import split_normal as sn

__all__ = [
    'PiEnsemble',
    'MvEnsemble'
]


class PiEnsemble(Estimator):
    """
    Ensemble of prediction interval estimators.
    """

    _METRICS = [picp, mpiw, cross, mse, pimse, within]
    _LOSS_THRESHOLD = 20.
    _RETRY_LIMIT = 10

    # PEP 526 — Syntax for Variable Annotations: <https://www.python.org/dev/peps/pep-0526/>

    def __init__(self, alpha, ensemble_size, aggreg_func, hidden_size, learning_rate, epochs,
                 batch_size, optimizer, loss_func, metrics=_METRICS, retry_limit=_RETRY_LIMIT,
                 retry_on_crossing=True, verbose=False, seed=None, output_dir=None, **kwargs):
        self._alpha = alpha
        self._ensemble_size = ensemble_size
        self._aggreg_func = aggreg_func
        self._hidden_size = hidden_size
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._metrics = metrics
        self._retry_limit = retry_limit
        self._retry_on_crossing = retry_on_crossing
        self._output_dir = output_dir
        self._verbose = verbose
        self._seed = seed
        self._kwargs = kwargs

        self._ensemble = []
        self._randomness = Randomness(self._seed)  # random state

    # PEP-3107 — Function Annotations: <https://www.python.org/dev/peps/pep-3107/>
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray) -> None:
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        x_val = torch.from_numpy(x_val)
        y_val = torch.from_numpy(y_val)

        i = 1
        retry_counter = 0
        while i <= self._ensemble_size:
            seed_ = self._randomness.random_seed()
            print('\n─ Model %d of %d [seed=%s]' % (i, self._ensemble_size, seed_))

            if self._output_dir:
                summary = dict()
                summary['train'] = SummaryWriter(self._output_dir + '/ensemble_%d/train' % seed_)
                summary['valid'] = SummaryWriter(self._output_dir + '/ensemble_%d/valid' % seed_)
            else:
                summary = None

            nn = PiNetwork(x_size=x_train.shape[1], y_size=3, h_size=self._hidden_size,
                           alpha=self._alpha, epochs=self._epochs, batch_size=self._batch_size,
                           learning_rate=self._learning_rate, optimizer=self._optimizer,
                           loss_func=self._loss_func, metrics=self._metrics, seed=seed_,
                           summary_writers=summary, **self._kwargs)
            nn.fit(x_train, y_train, x_val, y_val)

            if summary:
                summary['train'].close()
                summary['valid'].close()

            train_eval = nn.evaluate(x_train, y_train)
            if train_eval['loss'] > self._LOSS_THRESHOLD or (self._retry_on_crossing and (
                    train_eval.get('cross') and train_eval['cross'] > 0)):
                retry_counter += 1
                if retry_counter == self._retry_limit:
                    print('The optimization has not converged %d times and has reached the retry limit. No retries...' % retry_counter)
                    return False  # Fail
                print('The optimization has not converged. Retry...')
                continue

            self._ensemble.append(nn)
            retry_counter = 0
            i += 1

        return True  # Success

    def predict(self, x: np.ndarray, aggreg_func=None) -> np.ndarray:
        x = torch.from_numpy(x)
        y_pred_all = []
        for model in self._ensemble:
            y_pred = model.predict(x).cpu().numpy()
            y_pred_all.append(y_pred)
        y_pred_all = np.array(y_pred_all)
        aggregator = self._aggreg_func if aggreg_func is None else aggreg_func
        y_pred_l, y_pred_p, y_pred_u = aggregator(y_pred_all,
                                                  alpha=self._alpha,
                                                  seed=self._randomness.random_seed())
        return np.array([y_pred_l, y_pred_u, y_pred_p]).T

    def evaluate(self, x, y, y_transform_func=None, aggreg_func=None) -> np.ndarray:
        y_true = torch.from_numpy(y)
        y_pred = torch.from_numpy(self.predict(x, aggreg_func))

        if y_transform_func:
            y_true = y_transform_func(y_true)
            y_pred = y_transform_func(y_pred)

        # loss_all = []
        epochs_all = []
        for model in self._ensemble:
            # loss = model.loss(y_pred, y_true).detach().cpu().numpy()
            # loss_all.append(loss)
            epochs_all.append(model.actual_epochs)

        results = evaluate(y_pred, y_true, self._metrics)
        # results['loss'] = np.mean(loss_all)
        results['epochs'] = np.mean(epochs_all)

        # Convert to `numpy.ndarray`
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.numpy()

        return results

    def evaluate_models(self, x, y, y_transform_func=None) -> np.ndarray:
        results = []
        for model in self._ensemble:
            r = model.evaluate(torch.from_numpy(x), torch.from_numpy(y), y_transform_func)
            results.append(torch_to_numpy(r))
        return results

    def predict_models(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x)
        y_pred_all = []
        for model in self._ensemble:
            y_pred = model.predict(x).cpu().numpy()
            y_pred_all.append(y_pred)
        return np.array(y_pred_all)

    @staticmethod
    def load(file_path):
        return torch.load(file_path)

    def save(self, file_path):
        torch.save(self, file_path)

    @property
    def ensemble(self):
        return self._ensemble


class PiNetwork(Estimator):
    """
    Neural prediction interval estimator.
    """

    _METRICS = [picp, mpiw, cross, mse, pimse, within]

    def __init__(self, x_size, y_size, h_size, alpha, epochs, batch_size, learning_rate, optimizer,
                 loss_func, scheduler=None, early_stopping=False, punish_crossing=True,
                 metrics=_METRICS, seed=None, device=None, summary_writers=None, write_frequency=10,
                 print_frequency=100, **kwargs):
        self._x_size = x_size
        self._h_size = h_size
        self._y_size = y_size
        self._alpha = alpha
        self._epochs = epochs
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._scheduler = scheduler
        self._early_stopping = early_stopping
        self._loss_func = loss_func
        self._metrics = metrics
        self._seed = seed
        self._summary_writers = summary_writers
        self._write_frequency = write_frequency
        self._print_frequency = print_frequency
        self._punish_crossing = punish_crossing
        self._kwargs = kwargs

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)

        self._model = Dnn(x_size, h_size, y_size, seed=seed, **self._kwargs)
        self._model.to(self._device)
        self._randomness = Randomness(self._seed)

        self._actual_epochs = epochs

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        x_train = x_train.to(self._device)
        y_train = y_train.to(self._device)
        x_val = x_val.to(self._device)
        y_val = y_val.to(self._device)

        batcher = MiniBatches(x_train, y_train, self._batch_size, self._randomness)

        optimizer = self._optimizer(self._model.parameters(), lr=self._learning_rate)
        scheduler = None
        if self._scheduler:
            steps = batcher.batch_count * self._epochs
            scheduler = self._scheduler(optimizer, steps=steps, **self._kwargs)
        es = None
        if self._early_stopping:
            es = EarlyStopping(**self._kwargs)

        self._log_progress(0, optimizer, x_train, x_val, y_train, y_val)
        for epoch in range(1, self._epochs + 1):
            mini_batches = batcher.get_batches()
            for x, y in mini_batches:
                self._train_step(optimizer, scheduler, x, y)
            self._log_progress(epoch, optimizer, x_train, x_val, y_train, y_val)
            if es:
                y_pred = self._model(x_val)
                val_loss = self.loss(y_pred, y_val)
                if self._punish_crossing and cross(**y_to_kwargs(y_pred)) > 0.:
                    val_loss = float('inf')
                l_rate = optimizer.param_groups[0]['lr']
                es(epoch, val_loss, self._model, lr=l_rate)
                if es.early_stop:
                    print('Early stopping...')
                    break

        if es and es.early_stop:
            print('Loading state from epoch %d...' % es.epoch)
            self._actual_epochs = es.epoch
            self._model = es.load_checkpoint(self._model)
            es.clean()

    @property
    def actual_epochs(self):
        return self._actual_epochs

    def _log_progress(self, epoch, optimizer, x_train, x_val, y_train, y_val):
        validate = x_val is not None and y_val is not None
        write_summary = False if not self._write_frequency else \
            epoch % self._write_frequency == 0 or epoch == self._epochs
        print_summary = False if not self._print_frequency else \
            epoch % self._print_frequency == 0 or epoch == self._epochs
        if write_summary or print_summary:
            metrics_train = self.evaluate(x_train, y_train)
            metrics_val = self.evaluate(x_val, y_val) if validate else None
            if write_summary:
                self._write_summary(epoch, metrics_train, self._summary_writers, 'train')
                self._write_summary(epoch, metrics_val, self._summary_writers, 'valid')
            if print_summary:
                l_rate = optimizer.param_groups[0]['lr']
                self._print_progress(epoch, metrics_train, l_rate, prefix='train')
                self._print_progress(epoch, metrics_val, l_rate, prefix='valid')

    @staticmethod
    def _write_summary(epoch, metrics, summary, key):
        if summary and metrics:
            with summary[key] as writer:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, epoch)

    @staticmethod
    def _print_progress(epoch, metrics, l_rate, prefix):
        if metrics:
            print('ep: %4d  %s' % (epoch, prefix), end='  ')
            for key, value in metrics.items():
                print('%s %.6f' % (key, value), end='  ')
                # self._ex.log_scalar(prefix + '__' + key, value.numpy(), step=epoch)  # Sacred
            print('lr %.6f' % round(l_rate, 6), end='\n')

    def _train_step(self, optimizer, scheduler, x, y):
        optimizer.zero_grad()
        y_pred = self._model(x)
        loss = self._loss_func(y_pred, y, alpha=self._alpha, **self._kwargs)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    def _evaluate(self, y_pred, y_true):
        results = evaluate(y_pred, y_true, self._metrics)
        results['loss'] = self.loss(y_pred, y_true)
        return results

    def loss(self, y_pred, y_true):
        y_pred = y_pred.to(self._device)
        y_true = y_true.to(self._device)
        return self._loss_func(y_pred, y_true, alpha=self._alpha, **self._kwargs)

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        with torch.no_grad():
            x = x.to(self._device)
            y_pred = self._model(x)
        return y_pred

    def evaluate(self, x, y, y_transform_func=None):
        x = x.to(self._device)
        y = y.to(self._device)

        with torch.no_grad():
            y_pred = self._model(x)

        if y_transform_func:
            y = y_transform_func(y)
            y_pred = y_transform_func(y_pred)

        results = evaluate(y_pred, y, self._metrics)
        results['loss'] = self.loss(y_pred, y)
        results['epochs'] = self._actual_epochs
        return results


class Dnn(torch.nn.Module):

    def __init__(self, x_size, h_size, y_size, seed=None, **kwargs):
        super().__init__()
        self._params = self._initialize_parameters(x_size, h_size, y_size, seed)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise ValueError('Unknown data type. Expected numpy.ndarray or torch.Tensor instances.')
        return self._forward_propagation(x, self._params)

    def _initialize_parameters(self, x_size, h_size, y_size, seed=None):
        if seed:
            torch.manual_seed(seed)
        W = []
        b = []
        # input layer
        W.append(self._create_w((x_size, h_size[0]), 'W_0'))
        b.append(self._create_b((1, h_size[0]), 'b_0'))
        # hidden layers
        h_number = len(h_size)
        for i in range(1, h_number):
            W.append(self._create_w((h_size[i - 1], h_size[i]), 'W_%d' % i))
            b.append(self._create_b((1, h_size[i]), 'b_%d' % i))
        # output layer
        W.append(self._create_w((h_size[-1], y_size), 'W_%d' % h_number))
        b.append(self._create_b((1, y_size), 'b_%d' % h_number))
        return {'W': W, 'b': b}

    @staticmethod
    def _forward_propagation(x, parameters):
        W = parameters['W']
        b = parameters['b']
        # input layer
        z = torch.mm(x, W[0]) + b[0]
        a = torch.nn.functional.relu(z)
        # hidden layers (network)
        for i in range(1, len(W) - 1):
            z = torch.mm(a, W[i]) + b[i]
            a = torch.nn.functional.relu(z)
        # output layer
        z = torch.mm(a, W[-1]) + b[-1]
        return z

    def _create_w(self, dimensions, name):
        w = torch.nn.Parameter(torch.Tensor(*dimensions))
        torch.nn.init.xavier_uniform_(w)
        self.register_parameter(name, w)
        return w

    def _create_b(self, dimensions, name):
        b = torch.nn.Parameter(torch.Tensor(*dimensions))
        torch.nn.init.zeros_(b)
        self.register_parameter(name, b)
        return b


class MvEnsemble(Estimator):
    """
    Ensemble of mean-variance estimators.
    """

    _METRICS = [picp, mpiw, mse]
    _LOSS_THRESHOLD = 20.
    _RETRY_LIMIT = 10

    # PEP 526 — Syntax for Variable Annotations: <https://www.python.org/dev/peps/pep-0526/>

    def __init__(self, alpha, ensemble_size, aggreg_func, hidden_size, learning_rate, epochs,
                 batch_size, optimizer, loss_func, epsilon, metrics=_METRICS,
                 retry_limit=_RETRY_LIMIT, verbose=False, seed=None, output_dir=None, **kwargs):
        self._alpha = alpha
        self._ensemble_size = ensemble_size
        self._aggreg_func = aggreg_func
        self._hidden_size = hidden_size
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._epsilon = epsilon
        self._metrics = metrics
        self._retry_limit = retry_limit
        self._output_dir = output_dir
        self._verbose = verbose
        self._seed = seed
        self._kwargs = kwargs

        self._ensemble = []
        self._randomness = Randomness(self._seed)  # random state

    # PEP-3107 — Function Annotations: <https://www.python.org/dev/peps/pep-3107/>
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
            y_val: np.ndarray) -> None:
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        x_val = torch.from_numpy(x_val)
        y_val = torch.from_numpy(y_val)

        i = 1
        retry_counter = 0
        while i <= self._ensemble_size:
            seed_ = self._randomness.random_seed()
            print('\n─ Model %d of %d [seed=%s]' % (i, self._ensemble_size, seed_))

            if self._output_dir:
                summary = dict()
                summary['train'] = SummaryWriter(self._output_dir + '/ensemble_%d/train' % seed_)
                summary['valid'] = SummaryWriter(self._output_dir + '/ensemble_%d/valid' % seed_)
            else:
                summary = None

            nn = MvNetwork(x_size=x_train.shape[1], y_size=2, h_size=self._hidden_size,
                           alpha=self._alpha, epochs=self._epochs, batch_size=self._batch_size,
                           learning_rate=self._learning_rate, optimizer=self._optimizer,
                           loss_func=self._loss_func, epsilon=self._epsilon, seed=seed_,
                           summary_writers=summary, **self._kwargs)
            nn.fit(x_train, y_train, x_val, y_val)

            if summary:
                summary['train'].close()
                summary['valid'].close()

            train_eval = nn.evaluate(x_train, y_train)
            if train_eval['loss'] > self._LOSS_THRESHOLD:
                retry_counter += 1
                if retry_counter == self._retry_limit:
                    print('The optimization hasn\'t converged %d times and has reached the retry limit. No retries...' % retry_counter)
                    return False  # Fail
                print('The optimization hasn\'t converged. Retry...')
                continue

            self._ensemble.append(nn)
            retry_counter = 0
            i += 1

        return True  # Success

    def predict(self, x: np.ndarray, aggreg_func=None) -> np.ndarray:
        x = torch.from_numpy(x)
        y_pred_all = []
        for model in self._ensemble:
            y_pred = model.predict_(x).cpu().numpy()
            y_pred_all.append(y_pred)
        y_pred_all = np.array(y_pred_all)
        aggregator = self._aggreg_func if aggreg_func is None else aggreg_func
        y_pred_l, y_pred_p, y_pred_u = aggregator(y_pred_all,
                                                  alpha=self._alpha,
                                                  seed=self._randomness.random_seed())
        return np.array([y_pred_l, y_pred_u, y_pred_p]).T

    def evaluate(self, x, y, y_transform_func=None, aggreg_func=None) -> np.ndarray:
        """
        Returns results of the defined metrics and the mean loss.
        """
        y_true = torch.from_numpy(y)
        y_pred = torch.from_numpy(self.predict(x, aggreg_func))

        if y_transform_func:
            y_true = y_transform_func(y_true)
            y_pred = y_transform_func(y_pred)

        results = evaluate(y_pred, y_true, self._metrics)
        epochs_all = []
        for model in self._ensemble:
            epochs_all.append(model.actual_epochs)
        results['epochs'] = np.mean(epochs_all)

        # Convert to `numpy.ndarray`
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.numpy()

        return results

    def evaluate_models(self, x, y, y_transform_func=None) -> np.ndarray:
        results = []
        for model in self._ensemble:
            r = model.evaluate(torch.from_numpy(x), torch.from_numpy(y), y_transform_func)
            results.append(torch_to_numpy(r))
        return results

    def predict_models(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x)
        y_pred_all = []
        for model in self._ensemble:
            y_pred = model.predict(x).cpu().numpy()
            y_pred_all.append(y_pred)
        return np.array(y_pred_all)

    @staticmethod
    def load(file_path):
        return torch.load(file_path)

    def save(self, file_path):
        torch.save(self, file_path)

    @property
    def ensemble(self):
        return self._ensemble


class MvNetwork(Estimator):
    """
    Neural mean-variance estimator.
    """

    _METRICS = [mse, picp, mpiw]

    def __init__(self, x_size, y_size, h_size, alpha, epochs, batch_size, learning_rate, optimizer,
                 loss_func, epsilon, scheduler=None, early_stopping=False, metrics=_METRICS,
                 seed=None, device=None, summary_writers=None, write_frequency=10,
                 print_frequency=100, **kwargs):
        self._x_size = x_size
        self._h_size = h_size
        self._y_size = y_size
        self._alpha = alpha
        self._epochs = epochs
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._scheduler = scheduler
        self._early_stopping = early_stopping
        self._loss_func = loss_func
        self._epsilon = epsilon
        self._metrics = metrics
        self._seed = seed
        self._summary_writers = summary_writers
        self._write_frequency = write_frequency
        self._print_frequency = print_frequency
        self._kwargs = kwargs

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)

        self._model = MvDnn(x_size, h_size, y_size, seed=seed, **self._kwargs)
        self._model.to(self._device)
        self._randomness = Randomness(self._seed)

        self._actual_epochs = epochs

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        x_train = x_train.to(self._device)
        y_train = y_train.to(self._device)
        x_val = x_val.to(self._device)
        y_val = y_val.to(self._device)

        batcher = MiniBatches(x_train, y_train, self._batch_size, self._randomness)

        optimizer = self._optimizer(self._model.parameters(), lr=self._learning_rate)
        scheduler = None
        if self._scheduler:
            steps = batcher.batch_count * self._epochs
            scheduler = self._scheduler(optimizer, steps=steps, **self._kwargs)
        es = None
        if self._early_stopping:
            es = EarlyStopping(**self._kwargs)

        self._log_progress(0, optimizer, x_train, x_val, y_train, y_val)
        for epoch in range(1, self._epochs + 1):
            mini_batches = batcher.get_batches()
            for x, y in mini_batches:
                self._adversarial_train_step(optimizer, scheduler, x, y, self._epsilon)
            self._log_progress(epoch, optimizer, x_train, x_val, y_train, y_val)
            if es:
                y_pred = self._model(x_val)
                val_loss = self.loss(y_pred, y_val)
                l_rate = optimizer.param_groups[0]['lr']
                es(epoch, val_loss, self._model, lr=l_rate)
                if es.early_stop:
                    print('Early stopping...')
                    break

        if es and es.early_stop:
            print('Loading state from epoch %d...' % es.epoch)
            self._actual_epochs = es.epoch
            self._model = es.load_checkpoint(self._model)
            es.clean()

    @property
    def actual_epochs(self):
        return self._actual_epochs

    def _log_progress(self, epoch, optimizer, x_train, x_val, y_train, y_val):
        validate = x_val is not None and y_val is not None
        write_summary = False if not self._write_frequency else \
            epoch % self._write_frequency == 0 or epoch == self._epochs
        print_summary = False if not self._print_frequency else \
            epoch % self._print_frequency == 0 or epoch == self._epochs
        if write_summary or print_summary:
            metrics_train = self.evaluate(x_train, y_train)
            metrics_val = self.evaluate(x_val, y_val) if validate else None
            if write_summary:
                self._write_summary(epoch, metrics_train, self._summary_writers, 'train')
                self._write_summary(epoch, metrics_val, self._summary_writers, 'valid')
            if print_summary:
                l_rate = optimizer.param_groups[0]['lr']
                self._print_progress(epoch, metrics_train, l_rate, prefix='train')
                self._print_progress(epoch, metrics_val, l_rate, prefix='valid')

    @staticmethod
    def _write_summary(epoch, metrics, summary, key):
        if summary and metrics:
            with summary[key] as writer:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, epoch)

    @staticmethod
    def _print_progress(epoch, metrics, l_rate, prefix):
        if metrics:
            print('ep: %4d  %s' % (epoch, prefix), end='  ')
            for key, value in metrics.items():
                print('%s %.6f' % (key, value), end='  ')
                # self._ex.log_scalar(prefix + '__' + key, value.numpy(), step=epoch)  # Sacred
            print('lr %.6f' % round(l_rate, 6), end='\n')

    def _train_step(self, optimizer, scheduler, x, y):
        optimizer.zero_grad()
        y_pred = self._model(x)
        loss = self._loss_func(y_pred, y, alpha=self._alpha, **self._kwargs)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    def _fgsm_attack(self, x, epsilon, x_grad):
        sign_data_grad = x_grad.sign()
        perturbed_x = x + epsilon * sign_data_grad
        return perturbed_x

    def _adversarial_train_step(self, optimizer, scheduler, x, y, epsilon):
        if epsilon is not None and epsilon > 0.:
            optimizer.zero_grad()
            x.requires_grad = True
            y_pred = self._model(x)
            loss = self._loss_func(y_pred, y, alpha=self._alpha, **self._kwargs)
            loss.backward(retain_graph=True)

            # See https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
            x_grad = x.grad.data
            perturbed_x = self._fgsm_attack(x, epsilon, x_grad)

            optimizer.zero_grad()
            y_pred = self._model(x)
            perturbed_y_pred = self._model(perturbed_x)
            loss = self._loss_func(y_pred, y, alpha=self._alpha, **self._kwargs) + \
                   self._loss_func(perturbed_y_pred, y, alpha=self._alpha, **self._kwargs)
            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()
        else:
            self._train_step(optimizer, scheduler, x, y)

    def _evaluate(self, y_pred, y_true):
        results = evaluate(y_pred, y_true, self._metrics)
        results['loss'] = self.loss(y_pred, y_true)
        return results

    def loss(self, y_pred, y_true):
        y_pred = y_pred.to(self._device)
        y_true = y_true.to(self._device)
        return self._loss_func(y_pred, y_true, alpha=self._alpha, **self._kwargs)

    def predict_(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        with torch.no_grad():
            x = x.to(self._device)
            y_pred = self._model(x)
        return y_pred

    def predict(self, x):
        y_pred = self.predict_(x)
        y_sigma_1 = y_sigma_2 = y_pred[:, 0].cpu()
        y_mu = y_pred[:, 1].cpu()
        y_pred_pi = torch.zeros(y_pred.shape[0], 3)
        y_pred_pi[:, 0] = torch.tensor(sn.ppf(self._alpha / 2, y_mu, y_sigma_1, y_sigma_2))
        y_pred_pi[:, 1] = torch.tensor(sn.ppf(1 - self._alpha / 2, y_mu, y_sigma_1, y_sigma_2))
        y_pred_pi[:, 2] = y_mu
        return y_pred_pi.to(y_pred.device)

    def evaluate(self, x, y, y_transform_func=None):
        x = x.to(self._device)
        y = y.to(self._device)

        y_pred = self.predict(x)
        y_pred_ = self.predict_(x)

        if y_transform_func:
            y = y_transform_func(y)
            y_pred = y_transform_func(y_pred)

        results = evaluate(y_pred, y, self._metrics)
        results['loss'] = self.loss(y_pred_, y)
        results['epochs'] = self._actual_epochs
        return results


class MvDnn(Dnn):

    def __init__(self, x_size, h_size, y_size, seed=None, **kwargs):
        super().__init__(x_size, h_size, y_size, seed, **kwargs)

    def forward(self, x):
        z = self._forward_propagation(x, self._params)
        z[:, 0] = torch.nn.functional.softplus(z[:, 0].clone()) + 1e-6
        return z
