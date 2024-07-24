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
