#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Experiments with SEM- and SNM-QD+.
"""
from neural_pi.estimator.base import Randomness
from neural_pi.experiment import Experiment
from neural_pi.pipeline import DefaultPipeline
from neural_pi.data import FileDataset, ShuffledDataset
from neural_pi.estimator import PiEnsemble, Adam, ExponentialDecay, \
    qd_plus_loss, sem_aggreg, std_aggreg, snm_aggreg

EXPERIMENT_ID = 'exp_qdp2'
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
        hyper_params=dict(  # HPS trial number 50
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[50, 50],
            epochs=5,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.9974,
            lambda_2=0.51,
            ksi=10.,
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
        hyper_params=dict(  # HPS trial number 50
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[50, 50],
            epochs=662,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_rate=1.,
            decay_steps=50.,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.9974,
            lambda_2=0.51,
            ksi=10.,
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
            val_size=0.,
            test_size=0.1,
        ),
        num_runs=20,
        method=PiEnsemble,
        hyper_params=dict(  # HPS trial number 210
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.007,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=680,
            batch_size=100,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.9958,
            lambda_2=0.34,
            ksi=10.,
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
        hyper_params=dict(  # HPS trial number 278 and then manually fine-tuned
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.005,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=.99,
            epochs=1270,
            batch_size=100,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.9726,
            lambda_2=0.94,
            ksi=10.,
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
        hyper_params=dict(  # HPS trial number 77 and then manually fine-tuned
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.008,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=110,
            batch_size=100,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.9966,
            lambda_2=0.55,
            ksi=10.,
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
        hyper_params=dict(  # HPS trial number 21 and then manually fine-tuned
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.001,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.9975,
            epochs=1766,
            batch_size=100,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.9708,
            lambda_2=0.63,
            ksi=10.,
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
        hyper_params=dict(  # HPS trial number 73
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.008,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=211,
            batch_size=100,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.9964,
            lambda_2=0.09,
            ksi=10.,
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
        hyper_params=dict(  # HPS trial number 24 and then manually fine-tuned
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[100, 100],
            optimizer=Adam,
            learning_rate=0.02,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=80,
            batch_size=100,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.9976,
            lambda_2=0.24,
            ksi=10.,
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
        hyper_params=dict(  # HPS trial number 29 (handpicked)
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.008,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=660,
            batch_size=100,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.998,
            lambda_2=0.01,
            ksi=10.,
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
        hyper_params=dict(  # HPS trial number 14
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[50, 50],
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.95,
            epochs=980,
            batch_size=100,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.977,
            lambda_2=0.55,
            ksi=10.,
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
        hyper_params=dict(  # from HPS trial 77 and then manually fine-tuned
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg, snm_aggreg],
            hidden_size=[100, 100],
            optimizer=Adam,
            learning_rate=0.005,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=40,
            batch_size=1000,
            loss_func=qd_plus_loss,
            alpha=0.05,
            soften=160.,
            lambda_1=0.999,
            lambda_2=0.1,
            ksi=10.,
            print_frequency=1,
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
