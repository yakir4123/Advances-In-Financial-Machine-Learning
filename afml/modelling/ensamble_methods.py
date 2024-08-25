from typing import Any, Callable

import numpy as np
from sklearn.ensemble import BaggingClassifier as BaseBaggingClassifier
from sklearn.ensemble import RandomForestClassifier as BaseRandomForestClassifier
from sklearn.ensemble import _bagging, _forest

_original_forest_bootstrap_method = _forest._generate_sample_indices
_original_bagging_bootstrap_method = _bagging._generate_bagging_indices


class RandomForestClassifier(BaseRandomForestClassifier):

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        bootstrap_method: (
            Callable[[None | int | np.random.RandomState, int, int], np.ndarray] | None
        ) = None,
        **kwargs: Any
    ) -> None:
        super().__init__(n_estimators, **kwargs)
        self.bootstrap_method = bootstrap_method

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> None:
        if self.bootstrap and self.bootstrap_method is not None:
            # its nesty and probably may lead to bugs mostly on parallelism, but for learning purpose that's good enough
            _forest._generate_sample_indices = self.bootstrap_method
        try:
            super().fit(X, y, sample_weight)
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
        n_estimators: int = 100,
        *,
        bootstrap_method: Callable[[int], np.ndarray] | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(n_estimators, **kwargs)
        self.bootstrap_method = bootstrap_method

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> None:
        if self.bootstrap and self.bootstrap_method is not None:
            # its nesty and probably may lead to bugs mostly on parallelism, but for learning purpose that's good enough
            _bagging._generate_bagging_indices = (
                self._seq_bootstrap_generate_bagging_indices
            )
        try:
            super().fit(X, y, sample_weight)
        finally:
            if self.bootstrap and self.bootstrap_method is not None:
                _bagging._generate_bagging_indices = _original_bagging_bootstrap_method

    def _seq_bootstrap_generate_bagging_indices(
        self,
        random_state: None | int | np.random.RandomState,
        bootstrap_features: bool,
        bootstrap_samples: bool,
        n_features: int,
        n_samples: int,
        max_features: int,
        max_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly draw feature and sample indices."""
        # Get valid random state
        random_state = _bagging.check_random_state(random_state)

        # Draw indices
        feature_indices = _bagging._generate_indices(
            random_state, bootstrap_features, n_features, max_features
        )
        if bootstrap_samples and self.bootstrap_method is not None:
            sample_indices = self.bootstrap_method(max_samples)
        else:
            # the original bootstrap
            sample_indices = _bagging._generate_indices(
                random_state, bootstrap_samples, n_samples, max_samples
            )

        return feature_indices, sample_indices
