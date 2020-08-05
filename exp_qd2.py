#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Experiments with the original quality-driven ensembles (SEM-QD).
"""
from neural_pi.estimator.base import Randomness
from neural_pi.experiment import Experiment
from neural_pi.pipeline import DefaultPipeline
from neural_pi.data import FileDataset, ShuffledDataset
from neural_pi.estimator import PiEnsemble, Adam, ExponentialDecay, \
    qd_code_loss, sem_aggreg, std_aggreg

EXPERIMENT_ID = 'exp_qde2'
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
        hyper_params=dict(  # HPS trial number 241
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.004,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            epochs=5,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=40.,
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
        hyper_params=dict(  # HPS trial number 241
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.004,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            epochs=1042,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=40.,
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
        hyper_params=dict(  # HPS trial number 202
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.004,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=1184,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=23.,
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
        hyper_params=dict(  # HPS trial number 94
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.003,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=1415,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=4.,
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
        hyper_params=dict(  # HPS trial number 53
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.002,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=1355,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=28.,
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
        hyper_params=dict(  # HPS trial number 274
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.006,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=380,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=4.,
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
        hyper_params=dict(  # HPS trial number 80
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.002,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=995,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=11.,
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
        hyper_params=dict(  # HPS trial number 1
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[100, 100],
            optimizer=Adam,
            learning_rate=0.003,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=31,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=1,
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
            val_size=0.,
            test_size=0.1,
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 11
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.007,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=1606,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=40.,
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
            val_size=0.,
            test_size=0.1,
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 193
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.006,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=1214,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=1.,
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
            val_size=0.,
            test_size=0.1,
        ),
        num_runs=1,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 206
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[100, 100],
            optimizer=Adam,
            learning_rate=0.001,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.0,
            epochs=64,
            batch_size=1000,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=11.,
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
