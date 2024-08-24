from typing import Callable

import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.utils import check_random_state


@njit
def _num_co_events(
    t0: np.ndarray,
    t1: np.ndarray,
    close_idx: np.ndarray,
    iloc_start: int,
    iloc_end: int,
):
    count = np.zeros(iloc_end - iloc_start)
    for i in range(len(t0)):
        start_idx = np.searchsorted(close_idx, t0[i], side="left") - iloc_start
        end_idx = np.searchsorted(close_idx, t1[i], side="right") - iloc_start
        count[start_idx : end_idx] += 1
    return count


def num_co_events(close_idx: pd.Series, t1: pd.Series):
    """
    Snippet 4.1
    Compute the number of concurrent events per bar.
    """
    # 1) find events that span the period [molecule[0],molecule[-1]]
    t1 = t1.fillna(close_idx[-1])  # unclosed events still must impact other weights

    # 2) count events spanning a bar
    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    count_array = _num_co_events(
        t1.index.values, t1.values, close_idx.values, iloc[0], iloc[1]
    )
    return pd.Series(count_array, index=close_idx[iloc[0] : iloc[1] + 1])


@njit(parallel=True)
def _get_ind_matrix_fast(
    bars_indices: np.ndarray,
    events_starts: np.ndarray,
    events_ends: np.ndarray,
    n_events: int,
):
    n_bars = len(bars_indices)
    ind_m = np.zeros((n_bars, n_events))

    for i in prange(n_events):
        t0 = events_starts[i]
        t1 = events_ends[i]

        start_idx = np.searchsorted(bars_indices, t0, side="left")
        end_idx = np.searchsorted(bars_indices, t1, side="right")

        ind_m[start_idx:end_idx, i] = 1.0

    return ind_m


def get_ind_matrix(bars_indices: pd.Series, t1: pd.Series) -> pd.DataFrame:
    """
    Snippet 4.3 derives an indicator matrix from two arguments: the index of bars
    (bars_indices), and the pandas Series t1, which we used multiple times in Chapter 3. As
    a reminder, t1 is defined by an index containing the time at which the features are
    observed, and a values array containing the time at which the label is determined. The
    output of this function is a binary matrix indicating what (price) bars influence the
    label for each observation.
    """
    return pd.DataFrame(
        _get_ind_matrix_fast(bars_indices.values, t1.index.values, t1.values, len(t1)),
        index=bars_indices,
        columns=range(len(t1)),
    )


def get_avg_uniqueness(ind_m: pd.DataFrame) -> pd.Series:
    # Average uniqueness from indicator matrix
    c = ind_m.sum(axis=1)  # concurrency
    u = ind_m.div(c, axis=0)  # uniqueness
    avg_u = u[u > 0].mean()  # average uniqueness
    return avg_u


@njit
def _get_avg_uniqueness_fast(ind_m: np.ndarray, mean_index: int) -> np.ndarray:
    """
    Snippet 4.4 returns the average uniqueness of each observed feature. The input is
    the indicator matrix built by getIndMatrix
    """
    # NumPy array operations
    c = ind_m.sum(axis=1)
    mask = ind_m[:, 1] != 0
    u = ind_m[mask, mean_index] / c[mask]
    return u.mean()


@njit
def _choice(prob: np.ndarray) -> int:
    """
    numba not supporting choice function with custom prob
    """
    cumulative_prob = np.cumsum(prob)
    r = np.random.rand()
    return int(np.searchsorted(cumulative_prob, r, side="right"))


@njit
def _select_columns(ind_m: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected_columns = np.empty((ind_m.shape[0], len(indices)), dtype=ind_m.dtype)
    for i, idx in enumerate(indices):
        selected_columns[:, i] = ind_m[:, idx]
    return selected_columns


@njit(parallel=True)
def _seq_bootstrap_fast(ind_m: np.ndarray, s_length: int) -> np.ndarray:
    n_cols = ind_m.shape[1]
    phi = np.empty(s_length, dtype=np.int32)
    phi[0] = np.random.randint(0, n_cols)

    for j in range(1, s_length):
        avg_u = np.zeros(n_cols)
        for i in prange(n_cols):
            phi[j] = i
            ind_m_ = _select_columns(ind_m, phi[: j + 1])
            avg_u[i] = _get_avg_uniqueness_fast(ind_m_, -1)

        prob = avg_u / np.sum(avg_u)  # draw probabilities
        next_idx = _choice(prob)
        phi[j] = next_idx

    return phi


def bagging_seq_bootstrap(ind_m: pd.DataFrame) -> Callable[[int], np.ndarray]:
    """
    Snippet 4.5 gives us the index of the features sampled by sequential bootstrap. The
    inputs are the indicator matrix (indM) and an optional sample length (sLength), with
    a default value of as many draws as rows in indM
    """

    def bootstrap(n_samples: int):
        return _seq_bootstrap_fast(ind_m.values, n_samples)

    return bootstrap


def rf_seq_bootstrap(ind_m: pd.DataFrame) -> Callable[[None | int | np.random.RandomState, int, int], np.ndarray]:
    """
    Snippet 4.5 gives us the index of the features sampled by sequential bootstrap. The
    inputs are the indicator matrix (indM)
    """

    def bootstrap(random_state: None | int | np.random.RandomState, n_samples: int, n_samples_bootstrap: int):
        return _seq_bootstrap_fast(ind_m.values, n_samples_bootstrap)

    return bootstrap


@njit(parallel=True)
def _sample_w_by_uniqueness_fast(
    t0: np.ndarray,
    t1: np.ndarray,
    t_indices: np.ndarray,
    co_events: np.ndarray,
):
    """
    Snippet 4.2
    Given the co_weights by timestamps, it weights the sample
    """
    weight = np.zeros(len(t0))
    for i in prange(len(t0)):
        index_1 = np.searchsorted(t_indices, t0[i], side="left")
        index_2 = np.searchsorted(t_indices, t1[i], side="right")
        weight[i] = (1.0 / co_events[index_1:index_2]).mean()
    return weight


def sample_w_by_uniqueness(t1: pd.Series, co_events: pd.Series):
    """
    Snippet 4.2
    Given the co_weights by timestamps, it weights the sample
    """
    return pd.Series(
        _sample_w_by_uniqueness_fast(
            t1.index.values,
            t1.values,
            co_events.index.values,
            co_events.values,
        ),
        index=t1.index,
    )


@njit(parallel=True)
def _sample_w_by_return_fast(
    t0: np.ndarray,
    t1: np.ndarray,
    co_events_indices: np.ndarray,
    co_events: np.ndarray,
    log_ret: np.ndarray,
):
    """
    SNIPPET 4.10 determination of sample weight by absolute return attribution
    """
    # Derive sample weight by return attribution
    weights = np.zeros(len(t0))
    for i in prange(len(t0)):
        index_1 = np.searchsorted(co_events_indices, t0[i], side="left")
        index_2 = np.searchsorted(co_events_indices, t1[i], side="right")
        weights[i] = (log_ret[index_1:index_2] / co_events[index_1:index_2]).sum()
    return np.abs(weights)


def sample_w_by_return(t1: pd.DataFrame, co_events: pd.Series, close: pd.Series):
    """
    SNIPPET 4.10 determination of sample weight by absolute return attribution
    """
    ret = np.log(close).diff()  # log-returns, so that they are additive
    ret.iloc[0] = np.mean(ret)
    return pd.Series(
        _sample_w_by_return_fast(
            t1.index.values,
            t1.values,
            co_events.index.values,
            co_events.values,
            ret.values,
        ),
        index=t1.index,
    )


def linear_time_decay(tw, clf_last_w=1.0):
    """
    Snippet 4.11
    apply piecewise-linear decay to observed uniqueness (tw)
    the newest observation gets weight=1, oldest observation gets weight=clf_last_w
    """
    clf_w = tw.sort_index().cumsum()
    if clf_last_w >= 0:
        slope = (1.0 - clf_last_w) / clf_w.iloc[-1]
    else:
        slope = 1.0 / ((clf_last_w + 1) * clf_w.iloc[-1])
    const = 1.0 - slope * clf_w.iloc[-1]
    clf_w = const + slope * clf_w
    clf_w[clf_w < 0] = 0
    return clf_w


def exponential_time_decay(tw, exponent=2, clf_last_w=1.0):
    clf_w = tw.sort_index()
    clf_w -= clf_w.min()
    clf_w /= clf_w.max()
    clf_w = np.power(clf_w, exponent) / exponent * clf_last_w
    return clf_w



