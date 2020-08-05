import torch
import numpy as np
import functools
import multiprocessing as mp

from .base import Randomness
from . import split_normal as sn
from ..utils import rename

__all__ = [
    'mv_aggreg',
    'sem_aggreg',
    'std_aggreg',
    'snm_aggreg',
    'mean_aggreg',
    'no_aggreg',
    'normal_loss',
    'qd_code_loss',
    'qd_paper_loss',
    'qd_plus_loss',
    'mse_loss',
    'mse',
    'pimse',
    'cross',
    'positive',
    'within',
    'mpiw',
    'picp'
]


# --------------------------------------------------------------------------------------------------
# AGGREGATION METHODS
# --------------------------------------------------------------------------------------------------

def snm_aggreg(y_pred_all, alpha, seed=None, **kwargs):
    """
    Parallelization of the split normal aggregation.
    """
    print('\nAggregating. Have a tea... This may take a while depending on the dataset size.')
    return _parallelized_aggreg(alpha, y_pred_all, _split_normal_aggregator, seed)


def _split_normal_aggregator(alpha, y_pred, seed, *args):
    randomness = Randomness(seed)
    mixture_params = []
    for y_l, y_u, y_p in y_pred:  # Ensemble
        std_1, std_2 = sn.fit_split_normal(alpha=alpha,
                                           q_l=y_l, q_u=y_u, mode=y_p,
                                           seed=randomness.random_seed())
        mixture_params.append(dict(
            loc=y_p,
            scale_1=std_1,
            scale_2=std_2
        ))
    y_l_agg, y_p_agg, y_u_agg = _calc_y_from_sn_mixture(alpha, mixture_params)
    return y_p_agg, y_l_agg, y_u_agg


def mv_aggreg(y_pred_all, alpha, **kwargs):
    """
    Parallelizes the aggregation of results from normal estimators (MVE) to
    a normal mixture and calculates the desired PIs.
    """
    print('\nAggregating...')
    return _parallelized_aggreg(alpha, y_pred_all, _normal_aggregator)


def _normal_aggregator(alpha, y_pred, *args):
    """
    Aggregates ensembled normals as a normal mixture and calculates the desired PIs.

    Note: We use the split normal implementation since it is a generalization
    of the normal distribution.
    """
    mixture_params = []
    for y_sigma, y_mu in y_pred:  # Ensemble
        mixture_params.append(dict(
            loc=y_mu,
            scale_1=y_sigma,
            scale_2=y_sigma
        ))
    y_l_agg, y_p_agg, y_u_agg = _calc_y_from_sn_mixture(alpha, mixture_params)
    return y_p_agg, y_l_agg, y_u_agg


def _calc_y_from_sn_mixture(alpha, mixture_params):
    """
    Returns the final PI boundaries and point estimate given the `alpha` and
    mixture distribution parameters.
    """
    y_space = _estimate_snm_space(mixture_params)

    y_p_agg = 0
    mixture_pdf = None
    for pdf_params in mixture_params:  # Ensemble of distr. parameters
        p = sn.pdf(y_space, **pdf_params)
        y_p_agg += pdf_params['loc']
        if mixture_pdf is None:
            mixture_pdf = p
        else:
            mixture_pdf += p

    y_start = y_space[0]
    y_end = y_space[-1]
    num_samples = y_space.shape[0]
    ensemble_size = len(mixture_params)

    mixture_pdf = mixture_pdf / ensemble_size
    mixture_cdf = np.cumsum(mixture_pdf * (y_end - y_start) / num_samples)
    y_l_agg = sn.numerical_ppf_from_cdf(y_space, mixture_cdf, alpha / 2)
    y_u_agg = sn.numerical_ppf_from_cdf(y_space, mixture_cdf, 1 - alpha / 2)
    y_p_agg = y_p_agg / ensemble_size  # Point estimate

    return y_l_agg, y_p_agg, y_u_agg


def _estimate_snm_space(mixture_params, p_boundary=1e-6, step=1e-4):
    mixture_params = np.array(
        [[params[k] for k in ['loc', 'scale_1', 'scale_2']] for params in mixture_params])
    x_start = np.min(sn.ppf(p=p_boundary,
                            loc=mixture_params[:, 0],
                            scale_1=mixture_params[:, 1],
                            scale_2=mixture_params[:, 2]))
    x_end = np.max(sn.ppf(p=1 - p_boundary,
                          loc=mixture_params[:, 0],
                          scale_1=mixture_params[:, 1],
                          scale_2=mixture_params[:, 2]))
    num_samples = int((x_end - x_start) / step)
    return np.linspace(x_start, x_end, num_samples)


def _parallelized_aggreg(alpha, y_pred_all, aggreg_func, seed=None, **kwargs):
    """
    Parallelization of the prediction interval aggregation.
    """
    randomness = Randomness(seed)

    # y_pred_all.shape -> (sample, ensemble, {0, 1, 2})
    y_pred_all = np.swapaxes(y_pred_all, 0, 1)
    y_pred_l, y_pred_p, y_pred_u = np.zeros((3, y_pred_all.shape[0]), dtype=np.float32)

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(aggreg_func, [(alpha, y_pred, randomness.random_seed()) for y_pred in y_pred_all])
    pool.close()

    for i in range(y_pred_all.shape[0]):  # sample
        y_pred_l[i] = results[i][1]
        y_pred_u[i] = results[i][2]
        y_pred_p[i] = results[i][0]

    return y_pred_l, y_pred_p, y_pred_u


def sem_aggreg(y_pred_all, z_factor=1.96, **kwargs):
    """
    The original QDE aggregation function according to the implementation by
    Pearce et al.

    https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals
    """
    ddof = 1 if y_pred_all.shape[0] > 1 else 0
    y_pred_l = np.mean(y_pred_all[:, :, 0], axis=0) \
               - z_factor * np.std(y_pred_all[:, :, 0], axis=0, ddof=ddof) \
               / np.sqrt(y_pred_all.shape[0])  # !
    y_pred_u = np.mean(y_pred_all[:, :, 1], axis=0) \
               + z_factor * np.std(y_pred_all[:, :, 1], axis=0, ddof=ddof) \
               / np.sqrt(y_pred_all.shape[0])  # !
    y_pred_p = np.mean(y_pred_all[:, :, 2], axis=0)
    return y_pred_l, y_pred_p, y_pred_u


def std_aggreg(y_pred_all, z_factor=1.96, **kwargs):
    """
    The original QDE aggregation function according to the paper by
    Pearce et al.

    https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals
    """
    ddof = 1 if y_pred_all.shape[0] > 1 else 0
    y_pred_l = np.mean(y_pred_all[:, :, 0], axis=0) \
               - z_factor * np.std(y_pred_all[:, :, 0], axis=0, ddof=ddof)
    y_pred_u = np.mean(y_pred_all[:, :, 1], axis=0) \
               + z_factor * np.std(y_pred_all[:, :, 1], axis=0, ddof=ddof)
    y_pred_p = np.mean(y_pred_all[:, :, 2], axis=0)
    return y_pred_l, y_pred_p, y_pred_u


def mean_aggreg(y_pred_all, **kwargs):
    y_pred_l = np.mean(y_pred_all[:, :, 0], axis=0)
    y_pred_u = np.mean(y_pred_all[:, :, 1], axis=0)
    y_pred_p = np.mean(y_pred_all[:, :, 2], axis=0)
    return y_pred_l, y_pred_p, y_pred_u


def no_aggreg(y_pred_all, **kwargs):
    y_pred_l = y_pred_all[:, :, 0]
    y_pred_u = y_pred_all[:, :, 1]
    y_pred_p = y_pred_all[:, :, 2]
    return y_pred_l, y_pred_p, y_pred_u


# --------------------------------------------------------------------------------------------------
# LOSS FUNCTIONS
# --------------------------------------------------------------------------------------------------

def qd_plus_loss(y_pred, y_true, alpha, lambda_1, lambda_2, ksi, soften, epsilon=1e-3, **kwargs):
    y_l = y_pred[:, 0]  # lower bound
    y_u = y_pred[:, 1]  # upper bound
    y_p = y_pred[:, 2]  # point prediction
    y_t = y_true[:, 0]  # ground truth

    if soften is not None and soften > 0:
        # Soft: uses `sigmoid()`
        k_ = k_func_soft(y_l, y_t, y_u, soften)
        picp_ = picp_func_soft(y_l, y_t, y_u, soften)
    else:
        # Hard: uses `sign()`
        k_ = k_func(y_l, y_t, y_u)
        picp_ = picp_func(y_l, y_t, y_u)
    loss_mpiw_ = loss_mpiw(y_l, y_u, k_, epsilon)
    loss_picp_ = loss_picp(alpha, picp_)
    mse_ = torch.nn.functional.mse_loss(y_p, y_t)

    loss = (1 - lambda_1) * (1 - lambda_2) * loss_mpiw_ + \
           lambda_1 * (1 - lambda_2) * loss_picp_ + \
           lambda_2 * mse_ + ksi * penalty_func(y_l, y_p, y_u)

    return loss


def qd_loss(y_pred, y_true, loss_type, alpha, lambda_, soften, epsilon=1e-3, **kwargs):
    y_l = y_pred[:, 0]  # lower bound
    y_u = y_pred[:, 1]  # upper bound
    y_t = y_true[:, 0]  # ground truth

    n = torch.tensor(y_t.shape[0], dtype=torch.float).to(y_t.device)

    k_hard_ = k_func(y_l, y_t, y_u)  # Hard uses `sign()`
    loss_mpiw_hard_abs = loss_mpiw_abs(y_l, y_u, k_hard_, epsilon)
    loss_mpiw_hard = loss_mpiw(y_l, y_u, k_hard_, epsilon)
    picp_soft_ = picp_func_soft(y_l, y_t, y_u, soften)
    loss_picp_soft = loss_picp(alpha, picp_soft_)

    if loss_type == 'qd_code':
        # QD loss according to the code (Pearce et al., 2018)
        loss = loss_mpiw_hard_abs + lambda_ * torch.sqrt(n) * loss_picp_soft
    elif loss_type == 'qd_paper':
        # QD loss almost according to the paper (Pearce et al., 2018):
        # The paper contains `n / (alpha * (1 - alpha))` instead of `sqrt(n)` (from the accompanying
        # implementation). We have kept it this way so that we do not have to rescale the provided HPs.
        loss = loss_mpiw_hard + lambda_ * torch.sqrt(n) * loss_picp_soft
    else:
        raise ValueError('Unknown loss type.')

    return loss


qd_code_loss = functools.partial(qd_loss, loss_type='qd_code')
qd_paper_loss = functools.partial(qd_loss, loss_type='qd_paper')


def loss_mpiw(y_l: torch.Tensor, y_u: torch.Tensor, k: torch.Tensor,
              epsilon: float) -> torch.Tensor:
    return torch.div(torch.sum((y_u - y_l) * k), torch.sum(k) + epsilon)


def loss_mpiw_abs(y_l: torch.Tensor, y_u: torch.Tensor, k: torch.Tensor,
                  epsilon: float) -> torch.Tensor:
    return torch.div(torch.sum(torch.abs(y_u - y_l) * k), torch.sum(k) + epsilon)


def loss_picp(alpha: float, picp: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.relu((1. - alpha) - picp), 2)


def mse_loss(y_pred, y_true, **kwargs):
    y_p = y_pred[:, 2]  # point prediction
    y_t = y_true[:, 0]  # ground truth
    return torch.nn.functional.mse_loss(y_p, y_t)


def normal_loss(y_pred, y_true, **kwargs):
    """
    Negative log-likelihood (NLL) of normal distribution.
    """
    y_sigma = y_pred[:, 0]  # standard deviation
    y_mu = y_pred[:, 1]  # mean (point estimate)
    y_t = y_true[:, 0]  # ground truth

    a = torch.log(y_sigma)
    b = torch.div(torch.pow(y_t - y_mu, 2), 2 * torch.pow(y_sigma, 2))
    loss = torch.mean(a + b)

    return loss


# --------------------------------------------------------------------------------------------------
# MEASURES
# --------------------------------------------------------------------------------------------------

@rename('mse')
def mse_func(y_p: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Mean squared error.
    """
    return torch.mean((y_p - y) ** 2)


# alias
mse = mse_func


@rename('pimse')
def pimse_func(y_l: torch.Tensor, y_u: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
    y_p = torch.mean(torch.stack((y_l, y_u), axis=1), axis=1)
    return torch.mean(torch.pow(y - y_p, 2))


# alias
pimse = pimse_func


@rename('cross')
def cross_func(y_l: torch.Tensor, y_u: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.mean((y_l >= y_u).float())


# alias
cross = cross_func


@rename('positive')
def positive_func(y_l: torch.Tensor, y_u: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.mean(((y_l >= 0.) & (y_u >= 0.)).float())


# alias
positive = positive_func


@rename('within')
def within_func(y_l: torch.Tensor, y_u: torch.Tensor, y_p: torch.Tensor, **kwargs) -> torch.Tensor:
    above = y_l < y_p
    below = y_p < y_u
    within_bounds = torch.mul(above, below)
    return torch.mean(within_bounds.float())


# alias
within = within_func


def penalty_func(y_l: torch.Tensor, y_p: torch.Tensor, y_u: torch.Tensor) -> torch.Tensor:
    m_l = torch.relu(y_l - y_p)
    m_u = torch.relu(y_p - y_u)
    return torch.mean(m_l + m_u)


def k_func(y_l: torch.Tensor, y: torch.Tensor, y_u: torch.Tensor) -> torch.Tensor:
    """
    Returns a boolean mask showing whether item is inside the interval. Using `sign()`.
    """
    k_l = torch.relu(torch.sign(y - y_l))
    k_u = torch.relu(torch.sign(y_u - y))
    k = torch.mul(k_l, k_u)
    return k


def k_func_soft(y_l: torch.Tensor, y: torch.Tensor, y_u: torch.Tensor,
                soften: float) -> torch.Tensor:
    """
    Returns a boolean mask showing whether item is inside the interval. Using `sigmoid()`.
    """
    k_l = torch.sigmoid((y - y_l) * soften)
    k_u = torch.sigmoid((y_u - y) * soften)
    k = torch.mul(k_l, k_u)
    return k


@rename('picp')
def picp_func(y_l: torch.Tensor, y: torch.Tensor, y_u: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.mean(k_func(y_l, y, y_u))


# alias
picp = picp_func


def picp_func_soft(y_l: torch.Tensor, y: torch.Tensor, y_u: torch.Tensor,
                   soften: float) -> torch.Tensor:
    return torch.mean(k_func_soft(y_l, y, y_u, soften))


@rename('mpiw')
def mpiw_func(y_l: torch.Tensor, y_u: torch.Tensor, **kwargs):
    return torch.mean(y_u - y_l)


# alias
mpiw = mpiw_func
