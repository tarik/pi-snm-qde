#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Experiments for an ensemble of NNs optimized for MSE loss (point estimates).
"""
from neural_pi.estimator.base import Randomness
from neural_pi.experiment import Experiment
from neural_pi.pipeline import DefaultPipeline
from neural_pi.data import FileDataset, ShuffledDataset
from neural_pi.estimator import PiEnsemble, Adam, ExponentialDecay, mean_aggreg, mse_loss, mse

EXPERIMENT_ID = 'exp_mse2'
ex = Experiment(name=EXPERIMENT_ID,
                runs_dir='runs/%s' % EXPERIMENT_ID,
                temp_dir='temp',  # Where artifacts are temporarily stored.
                template='templates/template.html')


@ex.config
def default_config():
    executor = DefaultPipeline(ex.artifacts_dir)
    seed = Randomness().random_seed()


# ------------------------------------------------------------------------------

@ex.named_config
def boston_dev():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/boston_housing_data.csv',
            shuffle_path='data/boston_housing_data.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=5,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 227
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[50, 50],
            epochs=5,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.004,
            scheduler=ExponentialDecay,
            decay_rate=0.99,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )
    seed = 1


# ------------------------------------------------------------------------------

@ex.named_config
def boston():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/boston_housing_data.csv',
            shuffle_path='data/boston_housing_data.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 227
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[50, 50],
            epochs=129,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.004,
            scheduler=ExponentialDecay,
            decay_rate=0.99,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


@ex.named_config
def concrete():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/concrete_data.csv',
            shuffle_path='data/concrete_data.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 113
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[50, 50],
            epochs=300,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


@ex.named_config
def energy():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/energy.csv',
            shuffle_path='data/energy.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 8
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[50, 50],
            epochs=1099,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.01,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


@ex.named_config
def kin8():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/kin8nm.csv',
            shuffle_path='data/kin8nm.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 37
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[50, 50],
            epochs=91,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.01,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


@ex.named_config
def naval():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/naval_compressor_decay.csv',
            shuffle_path='data/naval_compressor_decay.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 99
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[50, 50],
            epochs=2420,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.001,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


@ex.named_config
def power():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/power.csv',
            shuffle_path='data/power.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 59
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[50, 50],
            epochs=847,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.006,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


@ex.named_config
def protein():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/protein.csv',
            shuffle_path='data/protein.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=5,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 6
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[100, 100],
            epochs=164,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.004,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


@ex.named_config
def wine():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/wine.csv',
            shuffle_path='data/wine.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 12
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[50, 50],
            epochs=10,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.01,
            scheduler=ExponentialDecay,
            decay_rate=0.95,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


@ex.named_config
def yacht():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/yacht.csv',
            shuffle_path='data/yacht.npy',
            standardize=True,
            shuffle=True
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 41
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[50, 50],
            epochs=798,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.01,
            scheduler=ExponentialDecay,
            decay_rate=0.99,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


@ex.named_config
def year():
    config = dict(
        dataset=FileDataset(
            file_path='data/yearmsd.csv',
            standardize=True,
            shuffle=False
        ),
        split=dict(
            train_size=0.9,
            test_size=0.1
        ),
        num_runs=1,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 47
            ensemble_size=5,
            aggreg_func=mean_aggreg,
            hidden_size=[100, 100],
            epochs=4,
            batch_size=1000,
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_rate=0.95,
            decay_steps=50.,
            loss_func=mse_loss,
            alpha=None,
            metrics=[mse],
            print_frequency=10,
            device='cpu'
        )
    )


# ------------------------------------------------------------------------------

@ex.capture
def execute(executor, config, seed):
    return executor.run(**config, seed=seed)


@ex.automain
def main():
    return execute()
