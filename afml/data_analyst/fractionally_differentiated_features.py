import numpy as np
import pandas as pd
from numba import njit, prange


@njit
def _get_weights_expand_window_fast(d: float, size: int) -> np.ndarray:
    """
    Snippet 5.1
    get_weights for fractional differentiation, generate weights for the time series
    to make it with enough memory and stationary.
    """
    w = np.empty(size, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = -w[k - 1] / k * (d - k + 1)
    w[np.isnan(w)] = 0
    w = np.ascontiguousarray(w[::-1]).reshape(-1, 1)
    return w


@njit
def _get_weights_ffd_fast(d, tau):
    w = [1.0]
    k = 1
    while abs(w[-1]) >= tau:
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    return np.array(w[::-1])


@njit
def _apply_weights_to_series_fast(w, series_values, l_star):
    result = np.full(series_values.shape[0], np.nan)
    for iloc in range(l_star, series_values.shape[0]):
        # result[iloc] = np.dot(w[-(iloc + 1):, 0], series_values[: iloc + 1])
        result[iloc] = np.dot(w[-(iloc + 1) :, 0].T, series_values[: iloc + 1])
    return result


def frac_diff_expanding_window(series: pd.DataFrame, d: float, tau: float = 0.01):
    """
    Snippet 5.2
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series
    # w = _get_weights_expand_window_fast(d, series.shape[0])
    w = _get_weights_expand_window_fast(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    l_star = w_[w_ > tau].shape[0]

    # 3) Apply weights to values
    df = pd.DataFrame(index=series.index)
    for name in series.columns:
        series_f = series[name].ffill().dropna()
        df[name] = _apply_weights_to_series_fast(w, series_f.values, l_star)
    return df


def getWeights(d, size):
    # thres>0 drops insignificant weights
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


@njit(parallel=True)
def _apply_weights(series, weights, width):
    n_rows = series.shape[0]
    n_cols = series.shape[1]
    result = np.empty((n_rows, n_cols))
    for col in prange(n_cols):
        series_col = series[:, col]
        for i in range(width, n_rows):
            if not np.isfinite(series_col[i]):
                result[i, col] = np.nan
                continue
            segment = np.ascontiguousarray(series_col[i - width : i + 1])
            result[i, col] = np.dot(weights, segment)
    return result


def frac_diff_ffd(series: pd.DataFrame, d: float, tau=1e-2):
    """
    Snippet 5.3
    Constant width window (new solution)
    Note 1: tau determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series
    w = _get_weights_ffd_fast(d, tau)
    width = len(w) - 1

    series_values = series.ffill().to_numpy()
    result = _apply_weights(series_values, w, width)

    # 4) Convert result back to pandas DataFrame
    result_df = pd.DataFrame(result, index=series.index, columns=series.columns)

    return result_df.iloc[width:]
