#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Experiments with the original quality-driven ensembles (SEM-QD) with 1 hidden layer NNs.
"""
from neural_pi.estimator.base import Randomness
from neural_pi.experiment import Experiment
from neural_pi.pipeline import DefaultPipeline
from neural_pi.data import FileDataset, ShuffledDataset
from neural_pi.estimator import PiEnsemble, Adam, ExponentialDecay, \
    qd_code_loss, qd_paper_loss, sem_aggreg, std_aggreg

EXPERIMENT_ID = 'exp_qd'
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            epochs=5,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.02,
            scheduler=ExponentialDecay,
            decay_rate=0.9,
            decay_steps=50.,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )
    seed = 1


# ------------------------------------------------------------------------------
# BASED ON THE IMPLEMENTATION: LOSS AND AGGREGATION FUNCTION
# ------------------------------------------------------------------------------

@ex.named_config
def boston_code():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            epochs=300,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.02,
            scheduler=ExponentialDecay,
            decay_rate=0.9,
            decay_steps=50.,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def concrete_code():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.03,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.98,
            epochs=800,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def energy_code():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.02,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.96,
            epochs=1000,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def kin8_code():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.02,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=500,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def naval_code():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.006,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.998,
            epochs=1000,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=4.,
            print_frequency=10
        )
    )


@ex.named_config
def power_code():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.01,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=300,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def protein_code():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[100],
            optimizer=Adam,
            learning_rate=0.002,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.999,
            epochs=600,
            batch_size=100,
            loss_func=qd_code_loss,
            retry_on_crossing=False,
            alpha=0.05,
            soften=160.,
            lambda_=40.,
            print_frequency=10
        )
    )


@ex.named_config
def wine_code():
    """
    Could not reproduce results of Pearce et al. (2018).
    Therefore new HPs with random search with the objective to optimize
    for the aggregation as did Pearce et al. (2018);
    confirmed in our email communication.
    """
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=700,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=26.,
            print_frequency=10
        )
    )


@ex.named_config
def yacht_code():
    """
    Original HPs with `alpha=0.01`.
    Therefore new HPs with random search with the objective to optimize
    for the aggregation as did Pearce et al. (2018);
    confirmed in our email communication.
    """
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.98,
            epochs=2500,
            batch_size=100,
            loss_func=qd_code_loss,
            alpha=0.05,
            soften=160.,
            lambda_=16.,
            print_frequency=10
        )
    )


@ex.named_config
def year_code():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[100],
            optimizer=Adam,
            learning_rate=0.005,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.999,
            epochs=100,
            batch_size=1000,
            loss_func=qd_code_loss,
            retry_on_crossing=False,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


# ------------------------------------------------------------------------------
# BASED ON THE PAPER: LOSS FUNCTION AND AGGREGATION FUNCTION
# ------------------------------------------------------------------------------

@ex.named_config
def boston_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            epochs=300,
            batch_size=100,
            optimizer=Adam,
            learning_rate=0.02,
            scheduler=ExponentialDecay,
            decay_rate=0.9,
            decay_steps=50.,
            loss_func=qd_paper_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def concrete_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.03,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.98,
            epochs=800,
            batch_size=100,
            loss_func=qd_paper_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def energy_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.02,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.96,
            epochs=1000,
            batch_size=100,
            loss_func=qd_paper_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def kin8_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.02,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=500,
            batch_size=100,
            loss_func=qd_paper_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def naval_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.006,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.998,
            epochs=1000,
            batch_size=100,
            loss_func=qd_paper_loss,
            alpha=0.05,
            soften=160.,
            lambda_=4.,
            print_frequency=10
        )
    )


@ex.named_config
def power_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.01,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.99,
            epochs=300,
            batch_size=100,
            loss_func=qd_paper_loss,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


@ex.named_config
def protein_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[100],
            optimizer=Adam,
            learning_rate=0.002,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.999,
            epochs=600,
            batch_size=100,
            loss_func=qd_paper_loss,
            retry_on_crossing=False,
            alpha=0.05,
            soften=160.,
            lambda_=40.,
            print_frequency=10
        )
    )


@ex.named_config
def wine_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=1.,
            epochs=700,
            batch_size=100,
            loss_func=qd_paper_loss,
            alpha=0.05,
            soften=160.,
            lambda_=26.,
            print_frequency=10
        )
    )


@ex.named_config
def yacht_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[50],
            optimizer=Adam,
            learning_rate=0.009,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.98,
            epochs=2500,
            batch_size=100,
            loss_func=qd_paper_loss,
            alpha=0.05,
            soften=160.,
            lambda_=16.,
            print_frequency=10
        )
    )


@ex.named_config
def year_paper():
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
        hyper_params=dict(
            ensemble_size=5,
            aggreg_func=[sem_aggreg, std_aggreg],
            hidden_size=[100],
            optimizer=Adam,
            learning_rate=0.005,
            scheduler=ExponentialDecay,
            decay_steps=50.,
            decay_rate=0.999,
            epochs=100,
            batch_size=1000,
            loss_func=qd_paper_loss,
            retry_on_crossing=False,
            alpha=0.05,
            soften=160.,
            lambda_=15.,
            print_frequency=10
        )
    )


# ------------------------------------------------------------------------------

@ex.capture
def execute(executor, config, seed):
    return executor.run(**config, seed=seed)


@ex.automain
def main():
    return execute()
