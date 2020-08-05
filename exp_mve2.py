#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Experiments with mean variance estimator (MVE) with adversarial training/examples (optional).
"""
from neural_pi.estimator.base import Randomness
from neural_pi.experiment import Experiment
from neural_pi.pipeline import DefaultPipeline
from neural_pi.data import ShuffledDataset, FileDataset
from neural_pi.estimator import MvEnsemble, Adam, ExponentialDecay, normal_loss, mv_aggreg

EXPERIMENT_ID = 'exp_mve2'
ex = Experiment(name=EXPERIMENT_ID,
                runs_dir='runs/%s' % EXPERIMENT_ID,
                temp_dir='temp',  # Where artifacts are temporarily stored.
                template='templates/template.html')


@ex.config
def default_config():
    executor = DefaultPipeline(ex.artifacts_dir)
    seed = Randomness().random_seed()


# --------------------------------------------------------------------------------------------------

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
            # train_size=0.81,  # validation split
            # test_size=0.09   # validation split
        ),
        num_runs=5,
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 40 (54 boston_1)
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[50, 50],
            epochs=5,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.005,
            # scheduler=ExponentialDecay,
            # decay_rate=1.,
            # decay_steps=50.,
            loss_func=normal_loss,
            epsilon=0.1,  # `None` to disable adversarial examples  # None (boston_1)
            alpha=0.05,
            # early_stopping=False,
            # patience=500,
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
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 40
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[50, 50],
            epochs=152,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.005,
            # scheduler=ExponentialDecay,
            # decay_rate=1.,
            # decay_steps=50.,
            loss_func=normal_loss,
            epsilon=0.1,  # `None` to disable adversarial examples
            alpha=0.05,
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
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 153
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[50, 50],
            epochs=66,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.0085,
            # scheduler=ExponentialDecay,
            # decay_rate=1.,
            # decay_steps=50.,
            loss_func=normal_loss,
            epsilon=0.04,  # `None` to disable adversarial examples
            alpha=0.05,
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
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 43
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[50, 50],
            epochs=313,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.01,
            scheduler=ExponentialDecay,
            decay_rate=.99,
            decay_steps=50.,
            loss_func=normal_loss,
            epsilon=None,  # `None` to disable adversarial examples
            alpha=0.05,
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
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 12
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[50, 50],
            epochs=31,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_rate=.99,
            decay_steps=50.,
            loss_func=normal_loss,
            epsilon=None,  # `None` to disable adversarial examples
            alpha=0.05,
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
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 33
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[50, 50],
            epochs=484,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.003,
            scheduler=ExponentialDecay,
            decay_rate=.99,
            decay_steps=50.,
            loss_func=normal_loss,
            epsilon=None,  # `None` to disable adversarial examples
            alpha=0.05,
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
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 24
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[50, 50],
            epochs=192,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.008,
            # scheduler=ExponentialDecay,
            # decay_rate=1.,
            # decay_steps=50.,
            loss_func=normal_loss,
            epsilon=None,  # `None` to disable adversarial examples
            alpha=0.05,
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
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 17
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[100, 100],
            epochs=83,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.006,
            # scheduler=ExponentialDecay,
            # decay_rate=1.,
            # decay_steps=500.,
            loss_func=normal_loss,
            epsilon=None,  # `None` to disable adversarial examples
            alpha=0.05,
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
            test_size=0.1
        ),
        num_runs=20,
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 25
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[50, 50],
            epochs=5,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.007,
            scheduler=ExponentialDecay,
            decay_rate=.96,
            decay_steps=500.,
            loss_func=normal_loss,
            epsilon=None,  # `None` to disable adversarial examples
            alpha=0.05,
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
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 38
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[50, 50],
            epochs=302,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_rate=.97,
            decay_steps=50.,
            loss_func=normal_loss,
            epsilon=None,  # `None` to disable adversarial examples
            alpha=0.05,
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
        method=MvEnsemble,
        hyper_params=dict(  # HPS trial number 2
            ensemble_size=5,
            aggreg_func=[mv_aggreg],
            hidden_size=[100, 100],
            epochs=4,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.004,
            scheduler=ExponentialDecay,
            decay_rate=.99,
            decay_steps=50.,
            loss_func=normal_loss,
            epsilon=None,  # `None` to disable adversarial examples
            alpha=0.05,
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
