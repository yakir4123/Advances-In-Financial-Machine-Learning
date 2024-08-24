from typing import Callable, Any

import numpy as np
from sklearn.ensemble import _forest, _bagging

from sklearn.ensemble import RandomForestClassifier as BaseRandomForestClassifier
from sklearn.ensemble import BaggingClassifier as BaseBaggingClassifier

_original_forest_bootstrap_method = _forest._generate_sample_indices
_original_bagging_bootstrap_method = _bagging._generate_bagging_indices


class RandomForestClassifier(BaseRandomForestClassifier):

    def __init__(
        self,
        n_estimators=100,
        *,
        bootstrap_method: (
            Callable[[None | int | np.random.RandomState, int, int], np.ndarray] | None
        ) = None,
        **kwargs
    ):
        super().__init__(n_estimators, **kwargs)
        self.bootstrap_method = bootstrap_method

    def fit(self, X: Any, y: Any, sample_weight: Any = None):
        if self.bootstrap and self.bootstrap_method is not None:
            # its nesty and probably may lead to bugs mostly on parallelism, but for learning purpose that's good enough
            _forest._generate_sample_indices = self.bootstrap_method
        try:
            super().fit()
        finally:
            if self.bootstrap and self.bootstrap_method is not None:
                _forest._generate_sample_indices = _original_forest_bootstrap_method


class BaggingClassifier(BaseBaggingClassifier):
    """
    Inherits the BaggingClassifier and add customize callback for bootstrap as scikit-learn doesn't support that. The
    default bootstrap is a static global method so to "override" this call the pointer to this method is modified
    before fit and return to default right after that.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        bootstrap_method: Callable[[int], np.ndarray] | None = None,
        **kwargs
    ):
        super().__init__(n_estimators, **kwargs)
        self.bootstrap_method = bootstrap_method

    def fit(self, X: Any, y: Any, sample_weight: Any = None):
        if self.bootstrap and self.bootstrap_method is not None:
            # its nesty and probably may lead to bugs mostly on parallelism, but for learning purpose that's good enough
            _bagging._generate_bagging_indices = (
                self._seq_bootstrap_generate_bagging_indices
            )
        try:
            super().fit()
        finally:
            if self.bootstrap and self.bootstrap_method is not None:
                _bagging._generate_bagging_indices = _original_bagging_bootstrap_method

    def _seq_bootstrap_generate_bagging_indices(
        self,
        random_state,
        bootstrap_features,
        bootstrap_samples,
        n_features,
        n_samples,
        max_features,
        max_samples,
    ):
        """Randomly draw feature and sample indices."""
        # Get valid random state
        random_state = _bagging.check_random_state(random_state)

        # Draw indices
        feature_indices = _bagging._generate_indices(
            random_state, bootstrap_features, n_features, max_features
        )
        if bootstrap_samples:
            sample_indices = self.bootstrap_method(max_samples)
        else:
            sample_indices = _bagging.sample_without_replacement(
                n_samples, n_samples, random_state=random_state
            )

        return feature_indices, sample_indices
