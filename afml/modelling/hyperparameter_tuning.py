from typing import Any, Literal

import numpy as np
import pandas as pd
from afml.modelling.cross_validation import PurgedKFold
from afml.modelling.ensamble_methods import BaggingClassifier
from scipy.stats import rv_continuous
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline


class SampleWeightedPipeline(Pipeline):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        **fit_params: Any,
    ) -> Pipeline:
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


def tune_hyper_params(
    features: pd.DataFrame,
    labels: pd.Series,
    t1: pd.Series,
    pipe_estimator: Any,
    param_grid: dict,
    cv: int = 3,
    search_type: Literal["grid", "random"] = "grid",
    n_jobs: int = -1,
    pct_embargo: float = 0.0,
    n_iter: int = 1,
    scoring: str = "f1",
    **fit_params: Any,
) -> BaseSearchCV:
    """
    SNIPPET 9.3 RANDOMIZED SEARCH WITH PURGED K-FOLD CV
    """
    # 1) hyperparameter search, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)  # purged
    match search_type:
        case "grid":
            search = GridSearchCV(
                estimator=pipe_estimator,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=n_jobs,
            )
        case "random":
            search = RandomizedSearchCV(
                estimator=pipe_estimator,
                param_distributions=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=n_jobs,
                n_iter=n_iter,
            )
        case _:
            raise ValueError("Unknown search method")
    return search.fit(features, labels, **fit_params)


def validate_model(
    search: BaseSearchCV,
    features: pd.DataFrame,
    labels: pd.Series,
    n_jobs: int = -1,
    n_estimators: int = 0,
    max_samples: float | None = None,
    max_features: float = 1.0,
    **fit_params: Any,
) -> Any:
    tune = search.best_estimator_  # pipeline
    # 2) fit validated model on the entirety of the data
    if max_samples is not None and max_samples > 0:
        tune = BaggingClassifier(
            estimator=SampleWeightedPipeline(tune.steps),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            n_jobs=n_jobs,
        )
        tune = tune.fit(
            features,
            labels,
            sample_weight=fit_params[
                tune.base_estimator.steps[-1][0] + "__sample_weight"
            ],
        )
        tune = Pipeline([("bag", tune)])
    return tune


class log_uniform_gen(rv_continuous):
    # random numbers log-uniformly distributed between 1 and e
    def _cdf(self, x: Any, *args: Any) -> float:
        return np.log(x / self.a) / np.log(self.b / self.a)


def log_uniform(a: float = 1, b: float = np.exp(1)) -> log_uniform_gen:
    return log_uniform_gen(a=a, b=b, name="log-uniform")
