#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Hyper-parameter search for mean variance estimator (MVE) with adversarial training/examples
(optional).
"""
from neural_pi.estimator.base import Randomness
from neural_pi.experiment import Experiment
from neural_pi.pipeline import CrossValidationPipeline, \
    HyperParameterSearch, LossObjective, PiObjective, HoldoutPipeline
from neural_pi.data import FileDataset, ShuffledDataset
from neural_pi.estimator import MvEnsemble, Adam, ExponentialDecay, \
    no_aggreg, mv_aggreg, normal_loss

EXPERIMENT_ID = 'hps_mve2'
ex = Experiment(name=EXPERIMENT_ID,
                runs_dir='runs/%s' % EXPERIMENT_ID,
                temp_dir='temp',  # Where artifacts are temporarily stored.
                template='templates/template.html')


@ex.config
def default_config():
    executor = HyperParameterSearch(
        pipeline_cls=CrossValidationPipeline,
        objective_cls=LossObjective,
        # objective_cls=PiObjective,
        output_dir=ex.artifacts_dir
    )
    seed = Randomness().random_seed()


# ------------------------------------------------------------------------------

@ex.named_config
def boston_dev():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/boston_housing_data.csv',
            shuffle_path='data/boston_housing_data.npy',
            standardize=True,
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[50, 50],
            epochs=500,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.0001, 0.01, 0.0001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=100,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=10
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
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[50, 50],
            epochs=2000,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.0001, 0.01, 0.0001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=500,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=200
    )


@ex.named_config
def concrete():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/concrete_data.csv',
            shuffle_path='data/concrete_data.npy',
            standardize=True,
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[50, 50],
            epochs=2000,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.0001, 0.01, 0.0001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=500,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=200
    )


@ex.named_config
def energy():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/energy.csv',
            shuffle_path='data/energy.npy',
            standardize=True,
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[50, 50],
            epochs=1000,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=50,
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=100,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=100
    )


@ex.named_config
def kin8():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/kin8nm.csv',
            shuffle_path='data/kin8nm.npy',
            standardize=True,
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[50, 50],
            epochs=1000,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=50,
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=100,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=100
    )


@ex.named_config
def naval():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/naval_compressor_decay.csv',
            shuffle_path='data/naval_compressor_decay.npy',
            standardize=True,
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[50, 50],
            epochs=1000,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=50,
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=100,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=100
    )


@ex.named_config
def power():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/power.csv',
            shuffle_path='data/power.npy',
            standardize=True,
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[50, 50],
            epochs=1000,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=50,
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=100,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=100
    )


@ex.named_config
def protein():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/protein.csv',
            shuffle_path='data/protein.npy',
            standardize=True,
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[100, 100],
            epochs=1000,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=50,
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=100,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=100
    )


@ex.named_config
def wine():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/wine.csv',
            shuffle_path='data/wine.npy',
            standardize=True,
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[50, 50],
            epochs=1000,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=50,
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=100,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=1000
    )


@ex.named_config
def yacht():
    config = dict(
        dataset=ShuffledDataset(
            data_path='data/yacht.csv',
            shuffle_path='data/yacht.npy',
            standardize=True,
            shuffle=0  # shuffle_id
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[50, 50],
            epochs=1000,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=50,
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=100,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=100
    )


@ex.named_config
def year():
    executor = HyperParameterSearch(
        pipeline_cls=HoldoutPipeline,
        objective_cls=PiObjective,
        output_dir=ex.artifacts_dir
    )
    config = dict(
        dataset=FileDataset(
            file_path='data/yearmsd.csv',
            standardize=True,
            shuffle=False
        ),
        split=dict(
            train_size=0.9,
        ),
        method=MvEnsemble,
        hyper_params=dict(
            ensemble_size=1,
            aggreg_func=no_aggreg,
            hidden_size=[100, 100],
            epochs=200,
            batch_size=100,
            optimizer=Adam,
            learning_rate=lambda t: t.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001),
            scheduler=ExponentialDecay,
            decay_rate=lambda t: t.suggest_discrete_uniform('decay_rate', 0.95, 1., 0.01),
            decay_steps=50,
            loss_func=normal_loss,
            epsilon=None,
            alpha=0.05,
            early_stopping=True,
            patience=50,
            delta=1e-6,
            tolerance=0.01,
            print_frequency=10,
            device='cpu'
        ),
        num_trials=100
    )


# ------------------------------------------------------------------------------

@ex.capture
def execute(executor, config, seed):
    return executor.run(**config, seed=seed)


@ex.automain
def main():
    return execute()
