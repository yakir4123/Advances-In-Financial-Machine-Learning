import numpy as np
import pandas as pd
from numba import njit, prange


def get_weights(d: float, size: int) -> np.ndarray:
    """
    Snippet 5.1
    get_weights for fractional differentiation, generate weights for the time series
    to make it with enough memory and stationary.
    """
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_expanding_window(series: pd.DataFrame, d: float, tau: float = .01):
    """
    Snippet 5.2
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series
    w = get_weights(d, series.shape[0])

    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > tau].shape[0]

    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        series_f, df_ = series[[name]].ffill().dropna(), pd.Series()
        for iloc in range(skip, series_f.shape[0]):
            loc = series_f.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


@njit
def get_weights_ffd(d, tau):
    w = [1.]
    k = 1
    while abs(w[-1]) >= tau:
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    return np.array(w[::-1])


def getWeights(d, size):
    # thres>0 drops insignificant weights
    w = [1.]
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
            segment = np.ascontiguousarray(series_col[i - width:i + 1])
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
    w = get_weights_ffd(d, tau)
    width = len(w) - 1

    # 2) Convert pandas DataFrame to numpy array for Numba
    series_values = series.ffill().to_numpy()

    # 3) Apply weights to values using Numba
    result = _apply_weights(series_values, w, width)

    # 4) Convert result back to pandas DataFrame
    result_df = pd.DataFrame(result, index=series.index, columns=series.columns)

    return result_df.iloc[width:]
