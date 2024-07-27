import numpy as np
import pandas as pd
from numba import njit


@njit
def _num_co_events(t1_index, t1_values, close_idx, iloc_start, iloc_end):
    count = np.zeros(iloc_end - iloc_start + 1)
    for tIn, tOut in zip(t1_index, t1_values):
        start_idx = np.searchsorted(close_idx, tIn, side='left') - iloc_start
        end_idx = np.searchsorted(close_idx, tOut, side='left') - iloc_start
        count[start_idx:end_idx + 1] += 1
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
    count_array = _num_co_events(t1.index.values, t1.values, close_idx.values, iloc[0], iloc[1])
    count = pd.Series(count_array, index=close_idx[iloc[0]:iloc[1] + 1])
    return count


def sample_tw(t1: pd.Series, co_events: pd.Series):
    """
    Snippet 4.2
    Given the co_weights by timestamps, it weights the sample
    """
    weight = pd.Series(index=t1.index)
    for tIn, tOut in t1.loc[weight.index].items():
        weight.loc[tIn] = (1. / co_events.loc[tIn:tOut]).mean()
    return weight


def linear_time_decay(tw, clf_last_w=1.):
    """
    Snippet 4.11
    apply piecewise-linear decay to observed uniqueness (tw)
    the newest observation gets weight=1, oldest observation gets weight=clf_last_w
    """
    clf_w = tw.sort_index().cumsum()
    if clf_last_w >= 0:
        slope = (1. - clf_last_w) / clf_w.iloc[-1]
    else:
        slope = 1. / ((clf_last_w + 1) * clf_w.iloc[-1])
    const = 1. - slope * clf_w.iloc[-1]
    clf_w = const + slope * clf_w
    clf_w[clf_w < 0] = 0
    return clf_w


def exponential_time_decay(tw, exponent=2, clf_last_w=1.):
    clf_w = tw.sort_index()
    clf_w -= clf_w.min()
    clf_w /= clf_w.max()
    clf_w = np.power(clf_w, exponent) / exponent * clf_last_w
    return clf_w


def get_ind_matrix(bars_indices: pd.Series, events: pd.Series) -> pd.DataFrame:
    """
    Snippet 4.3 derives an indicator matrix from two arguments: the index of bars
    (bars_indices), and the pandas Series t1, which we used multiple times in Chapter 3. As
    a reminder, t1 is defined by an index containing the time at which the features are
    observed, and a values array containing the time at which the label is determined. The
    output of this function is a binary matrix indicating what (price) bars influence the
    label for each observation.
    """
    ind_m = pd.DataFrame(0, index=bars_indices, columns=range(events.shape[0]))
    for i, (t0, t1) in enumerate(events.items()):
        ind_m.loc[t0:t1, i] = 1.
    return ind_m


def get_avg_uniqueness(indM: pd.DataFrame) -> pd.Series:
    """
    Snippet 4.4 returns the average uniqueness of each observed feature. The input is
    the indicator matrix built by getIndMatrix
    """
    # Average uniqueness from indicator matrix
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avg_u = u[u > 0].mean()  # average uniqueness
    return avg_u


def seqBootstrap(ind_m: pd.DataFrame, s_length=None):
    """
    Snippet 4.5 gives us the index of the features sampled by sequential bootstrap. The
    inputs are the indicator matrix (indM) and an optional sample length (sLength), with
    a default value of as many draws as rows in indM
    """
    if s_length is None:
        s_length = ind_m.shape[1]
    phi = []
    while len(phi) < s_length:
        avg_u = pd.Series()
        for i in ind_m:
            ind_m_ = ind_m[phi + [i]]  # reduce indM
            avg_u.loc[i] = get_avg_uniqueness(ind_m_).iloc[-1]
        prob = avg_u / avg_u.sum()  # draw prob
        phi += [np.random.choice(ind_m.columns, p=prob)]
    return phi
