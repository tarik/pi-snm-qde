import os
import optuna
import numpy as np
import pprint as pp
from types import LambdaType
from abc import ABC, abstractmethod
from sklearn.model_selection import ShuffleSplit, train_test_split

from .estimator.base import Randomness
from .utils import print_df, dict_to_dataframe, save_predictions, Timer


class DefaultPipeline:

    def __init__(self, output_dir):
        self._output_dir = output_dir

    def run(self, dataset, split, method, hyper_params, num_runs, seed):
        print('≡ Start [seed=%d]\n' % seed)
        pp.pprint(hyper_params)
        randomness = Randomness(seed)

        timer = Timer().start()

        aggreg_func = hyper_params['aggreg_func']
        if not isinstance(aggreg_func, list):
            aggreg_func = [aggreg_func]
        aggreg_std_measures = {f.__name__: [] for f in aggreg_func}

        all_abs_measures = []
        all_std_measures = []
        for r in range(1, num_runs + 1):
            r_shuffle = r - 1
            r_seed = randomness.random_seed()
            output_dir_ = self._output_dir + '/run_%s' % r_seed if self._output_dir else None
            print('\n═ Run %d of %d [seed=%s]' % (r, num_runs, r_seed))

            (x_train, y_train), _, (x_test, y_test) = dataset.load(**split, shuffle=r_shuffle, seed=r_seed)

            estimator = method(**hyper_params, seed=r_seed, output_dir=output_dir_)
            estimator.fit(x_train, y_train, x_test, y_test)

            # Saving non-standardized predictions and ground truth
            y_preds = estimator.predict_models(x_test)
            for i in range(len(y_preds)):
                y_preds[i] = dataset.inverse_y(y_preds[i])
            y_test_denorm = dataset.inverse_y(y_test)
            save_predictions(y_preds, y_test_denorm, r_shuffle, r_seed, self._output_dir)

            for af in aggreg_func:
                aggreg_std_measures[af.__name__].append(
                    estimator.evaluate(x_test, y_test, y_transform_func=dataset.standardize_y, aggreg_func=af)
                )

            # Non-standardized
            r_abs_measures = estimator.evaluate_models(x_test, y_test, dataset.inverse_y)
            # Standardized based on full dataset
            r_std_measures = estimator.evaluate_models(x_test, y_test, dataset.standardize_y)
            # Standardized based on training set
            for ram in r_abs_measures:
                all_abs_measures.append(ram)
            for rsm in r_std_measures:
                all_std_measures.append(rsm)

        timer.stop()

        print('\n═ Results:')
        if len(all_abs_measures) > 0:
            print('\n- Absolute:')
            self.print_summary(all_abs_measures)
            print('\n- Standardized:')
            report_df = self.print_summary(all_std_measures)
            # Aggregated (standardized)
            for af in aggreg_func:
                print('\n- Standardized `%s`:' % af.__name__)
                self.print_summary(aggreg_std_measures[af.__name__])
        else:
            print('No results probably due to unsuccessful training.')

        print('\n═ Duration: %.1f s (%s → %s)' % (np.round(timer.duration(), 1),
                                                  timer.start_time.strftime('%H:%M:%S'),
                                                  timer.stop_time.strftime('%H:%M:%S')))
        return report_df.T.to_dict()

    @staticmethod
    def print_summary(measures, func=['mean', 'std', 'sem'], round=6):
        # Unaggregated
        no_aggreg_df = dict_to_dataframe(measures)
        print_df(no_aggreg_df.T)
        # Aggregated
        aggreg_df = no_aggreg_df.agg(func, axis=1).round(round)  # ddof=1
        print_df(aggreg_df)
        return aggreg_df


# --------------------------------------------------------------------------------------------------

class HyperParameterSearch:

    def __init__(self, objective_cls, pipeline_cls, output_dir=None):
        self._sampler_cls = optuna.samplers.RandomSampler
        self._objective_cls = objective_cls
        self._pipeline_cls = pipeline_cls
        self._output_dir = output_dir
        self._study = None
        self._num_trials = None

    @staticmethod
    def _sort_trials_df(trials_df, by=[], ascending=True, inplace=True):
        by_ = [('user_attrs', name) for name in by]
        trials_df.sort_values(by=by_, ascending=ascending, inplace=inplace)
        return trials_df

    def run(self, dataset, split, method, hyper_params, num_trials, seed):
        sampler = self._sampler_cls(seed=seed)
        objective = self._objective_cls(self._pipeline_cls, dataset, split, method, hyper_params, seed)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=num_trials, n_jobs=1)

        trials_df = study.trials_dataframe()
        print_df(trials_df)

        if self._output_dir:
            trials_df.to_csv(self._output_dir + '/trials_%d.csv' % seed)

        self._study = study
        self._num_trials = num_trials

        return trials_df.to_dict()

    @property
    def study(self):
        return self._study


class Objective(ABC):

    def __init__(self, pipeline_cls, dataset, split, method, hyper_params, seed):
        self._pipeline_cls = pipeline_cls
        self._dataset = dataset
        self._split = split
        self._method = method
        self._hyper_params = hyper_params
        self._seed = seed

    def __call__(self, trial):
        print('=' * 100)
        print('Trial number %d' % trial.number)
        print('=' * 100, '\n')

        trial_hps = dict()
        for key, value in self._hyper_params.items():
            if isinstance(value, LambdaType) and str(value.__name__) == '<lambda>':
                trial_hps[key] = value(trial)
            else:
                trial_hps[key] = value

        for t in trial.study.trials:
            if t.state != optuna.structs.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.structs.TrialPruned('Duplicate hyper-parameters.')

        pipeline = self._pipeline_cls()
        trial_results = pipeline.run(self._dataset, self._split, self._method, trial_hps, self._seed)

        if trial_results is None:  # Indicates a failed trial
            return float("inf")

        print('\n', '=' * 100, '\n', sep='')

        for m in trial_results.keys():
            for a in trial_results[m].keys():
                trial.set_user_attr('%s_%s' % (m, a), trial_results[m][a])

        return self.evaluate(trial_results, trial_hps)

    @abstractmethod
    def evaluate(self, report, hyper_params):
        return NotImplemented


# class PiObjective(Objective):
#     """
#     Returns MPIW if PICP is within tolerance.
#     """
#
#     def evaluate(self, report, hyper_params):
#         alpha = hyper_params['alpha']
#         tolerance = hyper_params['tolerance']
#         picp = report['picp']['mean']
#         mpiw = report['mpiw']['mean']
#         apie = abs(1 - alpha - picp)  # Absolute prediction interval error
#         if apie > tolerance or mpiw <= 0:
#             return float("inf")
#         return mpiw


class PiObjective(Objective):
    """
    Returns MPIW or penalized MPIW based on PICP being within tolerance.
    """

    def evaluate(self, report, hyper_params):
        alpha = hyper_params['alpha']
        tolerance = hyper_params['tolerance']
        picp = report['picp']['mean']
        mpiw = report['mpiw']['mean']
        if mpiw <= 0:
            return float("inf")
        apie = abs(1 - alpha - picp)  # Absolute prediction interval error
        disparity = apie // tolerance
        return mpiw * (10 ** disparity)


class MseObjective(Objective):

    def evaluate(self, report, hyper_params):
        return report['mse']['mean']


class LossObjective(Objective):

    def evaluate(self, report, hyper_params):
        return report['loss']['mean']


# --------------------------------------------------------------------------------------------------

class CrossValidationPipeline:
    """
    Mainly for hyper-parameter search.
    """

    def __init__(self, output_dir=None):
        self._output_dir = output_dir

    def run(self, dataset, split, method, hyper_params, seed, val_size=0.1, n_splits=5):
        timer = Timer().start()

        randomness = Randomness(seed)
        seed_ = randomness.random_seed()
        output_dir_ = self._output_dir + '/run_%s' % seed_ if self._output_dir else None
        print('\n═ Running [seed=%s]' % (seed_))

        (x, y), _, _ = dataset.load(**split, shuffle=None, seed=seed_)
        rs = ShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=seed_)

        measures = []
        for train_ids, val_ids in rs.split(x):
            s_seed = randomness.random_seed()
            estimator = method(**hyper_params, seed=s_seed, output_dir=output_dir_)
            x_train, y_train = x[train_ids], y[train_ids]
            x_val, y_val = x[val_ids], y[val_ids]
            success = estimator.fit(x_train, y_train, x_val, y_val)
            if not success:
                return None
            measures += estimator.evaluate_models(x_val, y_val, dataset.standardize_y)

        print(measures)

        timer.stop()

        print('\n═ Results:\n')
        # Results per model in ensembles
        report_df = dict_to_dataframe(measures).T
        print_df(report_df)
        # Aggregated results
        report_df = dict_to_dataframe(measures)
        report_df = report_df.agg(['mean', 'std', 'sem'], axis=1).round(6)  # ddof=1
        print_df(report_df)

        print('\n═ Duration: %.1f s (%s → %s)' % (np.round(timer.duration(), 1),
                                                  timer.start_time.strftime('%H:%M:%S'),
                                                  timer.stop_time.strftime('%H:%M:%S')))
        return report_df.T.to_dict()


class HoldoutPipeline:
    """
    Mainly for hyper-parameter search.
    """

    def __init__(self, output_dir=None):
        self._output_dir = output_dir

    def run(self, dataset, split, method, hyper_params, seed, val_size=0.1):
        timer = Timer().start()

        randomness = Randomness(seed)
        seed_ = randomness.random_seed()
        output_dir_ = self._output_dir + '/run_%s' % seed_ if self._output_dir else None
        print('\n═ Running [seed=%s]' % (seed_))

        (x, y), _, _ = dataset.load(**split, shuffle=None, seed=seed_)  # `(x, y)` is a training set
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size, shuffle=False)

        measures = []
        estimator = method(**hyper_params, seed=seed_, output_dir=output_dir_)
        success = estimator.fit(x_train, y_train, x_val, y_val)
        if not success:
            return None
        measures += estimator.evaluate_models(x_val, y_val, dataset.standardize_y)

        print(measures)

        timer.stop()

        print('\n═ Results:\n')
        # Results per model in ensembles
        report_df = dict_to_dataframe(measures).T
        print_df(report_df)
        # Aggregated results
        report_df = dict_to_dataframe(measures)
        report_df = report_df.agg(['mean', 'std', 'sem'], axis=1).round(6)  # ddof=1
        print_df(report_df)

        print('\n═ Duration: %.1f s (%s → %s)' % (np.round(timer.duration(), 1),
                                                  timer.start_time.strftime('%H:%M:%S'),
                                                  timer.stop_time.strftime('%H:%M:%S')))
        return report_df.T.to_dict()
