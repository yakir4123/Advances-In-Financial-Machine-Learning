from typing import Any, Iterator

import numpy as np
import pandas as pd
from numba import njit
from sklearn.model_selection._split import _BaseKFold


def get_train_times(t1: pd.Series, testTimes: pd.Series) -> pd.Series:
    """
    SNIPPET 7.1-PURGING OBSERVATION IN THE TRAINING SET
    Given testTimes, find the times of the training observations.
    —t1.index: Time when the observation started.
    —t1.value: Time when the observation ended.
    —testTimes: Times of testing observations.
    """
    mask = _get_train_times_fast(
        t1.index.values, t1.values, testTimes.index.values, testTimes.values
    )
    return t1[mask]


def get_embargo_times(times: np.ndarray, pct_embargo: float) -> pd.Series:
    """
    SNIPPET 7.2 EMBARGO ON TRAINING OBSERVATIONS
    given times and percentage embargo, return series of embargo times
    """
    # Get embargo time for each bar
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        embargo = pd.Series(times, index=times)
    else:
        embargo = pd.Series(times[step:], index=times[:-step])
        embargo = embargo.append(pd.Series(times[-1], index=times[-step:]))
    return embargo


class PurgedKFold(_BaseKFold):
    """
    Snippet 7.3 - CROSS-VALIDATION CLASS WHEN OBSERVATION OVERLAP
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    """

    def __init__(self, n_splits: int, t1: pd.Series, pct_embargo: float = 0.0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pct_embargo)
        step = int(X.shape[0] / self.n_splits)
        test_starts = [[i, i + step] for i in range(0, X.shape[0] // step * step, step)]
        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            max_t1_idx = self.t1.index.searchsorted(self.t1[X.index[i:j]].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if max_t1_idx < X.shape[0]:
                train_indices = np.concatenate(
                    (train_indices, indices[max_t1_idx + embargo :])
                )
            yield train_indices, test_indices


@njit
def _get_train_times_fast(
    t0: np.ndarray, t1: np.ndarray, test_starts: np.ndarray, test_ends: np.ndarray
) -> np.ndarray:
    mask = np.ones(t0.shape[0], dtype=bool)

    for i in range(test_starts.shape[0]):
        test_start = test_starts[i]
        test_end = test_ends[i]

        mask &= ~((test_start <= t0) & (t0 <= test_end))  # train starts within test
        mask &= ~((test_start <= t1) & (t1 <= test_end))  # train ends within test
        mask &= ~((t0 <= test_start) & (test_end <= t1))  # train envelops test

    return mask
