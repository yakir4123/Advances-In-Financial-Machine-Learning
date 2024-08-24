from typing import Callable, Any
from sklearn.ensemble import _forest
from sklearn.ensemble import RandomForestClassifier as BaseRandomForestClassifier

_original_bootstrap_method = _forest._generate_sample_indices


class RandomForestClassifier(BaseRandomForestClassifier):

    def __init__(self, n_estimators=100, *, bootstrap_method: Callable | None = None, **kwargs):
        super().__init__(n_estimators, **kwargs)
        self.bootstrap_method = bootstrap_method

    def fit(self, X: Any, y: Any, sample_weight: Any = None):
        if self.bootstrap and self.bootstrap_method is not None:
            # its nesty and probably may lead to bugs mostly on parallelism, but for learning purpose that's good enough
            _forest._generate_sample_indices = bootstrap_method
        super().fit()
        if self.bootstrap and self.bootstrap_method is not None:
            _forest._generate_sample_indices = _original_bootstrap_method
