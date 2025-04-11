from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from afml.modelling.cross_validation import PurgedKFold, cv_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection._split import _BaseKFold
from sklearn.tree import DecisionTreeClassifier


def feature_importance_mdi(
    fitted_rf_classifier: RandomForestClassifier, feat_names: list[str]
) -> pd.DataFrame:
    """
    SNIPPET 8.2 MDI FEATURE IMPORTANCE
    * Fast In-Sample feature importance
    * With substitution effects
    """
    _df0 = {
        i: tree.feature_importances_
        for i, tree in enumerate(fitted_rf_classifier.estimators_)
    }
    df0 = pd.DataFrame.from_dict(_df0, orient="index")
    df0.columns = feat_names
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat(
        {"mean": df0.mean(), "std": df0.std() * df0.shape[0] ** -0.5}, axis=1
    )
    imp /= imp["mean"].sum()
    return imp


def feature_importance_mda(
    classifier: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    sample_weight: pd.Series,
    t1: pd.Series,
    pct_embargo: float,
    scoring: str = "neg_log_loss",
) -> tuple[pd.DataFrame, float]:
    """
    SNIPPET 8.3 MDA FEATURE IMPORTANCE
    * Slow Out Of Samples feature importance
    * With substitution effects
    """
    # feat importance based on OOS score reduction
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise Exception("wrong scoring method.")
    cvGen = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)  # purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = classifier.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(
                y1, prob, sample_weight=w1.values, labels=classifier.classes_
            )
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(
                    y1, prob, sample_weight=w1.values, labels=classifier.classes_
                )
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)
    imp = (-scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)
    imp = pd.concat(
        {"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1
    )
    return imp, scr0.mean()


def feature_importance_sfi(
    classifier: Any,
    X: pd.DataFrame,
    labels: pd.Series,
    sample_weight: pd.Series,
    feat_names: Sequence[str] | None = None,
    scoring: Literal["neg_log_loss", "accuracy"] = "neg_log_loss",
    cv_gen: _BaseKFold | None = None,
) -> pd.DataFrame:
    """
    SNIPPET 8.4 IMPLEMENTATION OF SFI
    * Out Of Sample feature importance
    * Check the importance of one feature at a time
    * Without substitution effects
    """
    imp = pd.DataFrame(columns=["mean", "std"])
    feat_names = feat_names or X.columns

    for feat_name in feat_names:
        df0 = cv_score(
            classifier,
            X=X[[feat_name]],
            y=labels["bin"],
            sample_weight=sample_weight,
            scoring=scoring,
            cv_gen=cv_gen,
        )
        imp.loc[feat_name, "mean"] = df0.mean()
        imp.loc[feat_name, "std"] = df0.std() * df0.shape[0] ** -0.5
    return imp


def _get_eigen_vector(dot: pd.DataFrame, var_thres: float) -> pd.DataFrame:
    # compute eVec from dot prod matrix, reduce dimension
    e_val, e_vec = np.linalg.eigh(dot)
    idx = e_val.argsort()[::-1]  # arguments for sorting eVal desc
    e_val, e_vec = e_val[idx], e_vec[:, idx]

    # 2) only positive eVals
    e_val = pd.Series(e_val, index=["PC_" + str(i + 1) for i in range(e_val.shape[0])])
    e_vec = pd.DataFrame(e_vec, index=dot.index, columns=e_val.index)
    e_vec = e_vec.loc[:, e_val.index]

    # 3) reduce dimension, form PCs
    cum_var = e_val.cumsum() / e_val.sum()
    dim = cum_var.values.searchsorted(var_thres)
    e_vec = e_vec.iloc[:, : dim + 1]
    return e_vec


def orthogonal_features(X: pd.DataFrame, var_threshold: float = 0.95) -> pd.DataFrame:
    """
    SNIPPET 8.5 COMPUTATION OF ORTHOGONAL FEATURES
    Partial solution to the substitution problem with MDI and MDA
    """
    # Given a dataframe dfX of features, compute orthogonal_features dfP
    df_z = X.sub(X.mean(), axis=1).div(X.std(), axis=1)  # standardize
    dot = pd.DataFrame(np.dot(df_z.T, df_z), index=X.columns, columns=X.columns)
    e_vec = _get_eigen_vector(dot, var_threshold)
    df_p = np.dot(df_z, e_vec)
    return df_p


def feat_importance(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    n_estimators: int = 1000,
    n_splits: int = 10,
    max_samples: int = 1,
    pct_embargo: float = 0,
    scoring: Literal["neg_log_loss", "accuracy"] = "accuracy",
    method: Literal["MDI", "MDA", "SFI"] = "SFI",
    min_w_leaf: float = 0.0,
    max_features: int | None = 1,
    n_jobs: int = 1,
    **kwargs: Any
) -> tuple[pd.DataFrame, float, float]:
    # feature importance from a random forest
    # 1) prepare classifier,cv. max_features=1, to prevent masking
    classifier = DecisionTreeClassifier(
        criterion="entropy",
        max_features=max_features,
        class_weight="balanced",  # type: ignore
        min_weight_fraction_leaf=min_w_leaf,
    )
    classifier = BaggingClassifier(
        estimator=classifier,
        n_estimators=n_estimators,
        max_features=1.0,
        max_samples=max_samples,
        oob_score=True,
        n_jobs=n_jobs,
    )
    fit = classifier.fit(X=X, y=labels["bin"], sample_weight=labels["w"].values)
    oob = fit.oob_score_  # type: ignore
    if method == "MDI":
        imp = feature_importance_mdi(fit, feat_names=list(X.columns))  # type: ignore
        oos = cv_score(
            classifier,
            X=X,
            y=labels["bin"],
            n_splits=n_splits,
            sample_weight=labels["w"],
            t1=labels["t1"],
            pct_embargo=pct_embargo,
            scoring=scoring,
        ).mean()
    elif method == "MDA":
        imp, oos = feature_importance_mda(
            classifier,
            X=X,
            y=labels["bin"],
            n_splits=n_splits,
            sample_weight=labels["w"],
            t1=labels["t1"],
            pct_embargo=pct_embargo,
            scoring=scoring,
        )
    elif method == "SFI":
        cv_gen = PurgedKFold(
            n_splits=n_splits, t1=labels["t1"], pct_embargo=pct_embargo
        )
        oos = cv_score(
            classifier,
            X=X,
            y=labels["bin"],
            sample_weight=labels["w"],
            scoring=scoring,
            cv_gen=cv_gen,
        ).mean()
        imp = feature_importance_sfi(
            classifier=classifier,
            X=X,
            labels=labels,
            sample_weight=labels["w"],
            scoring=scoring,
            cv_gen=cv_gen,
        )
    else:
        raise ValueError(
            'Undefined feature importance. Only valid features importance are ["MDI", "MDA", "SFI"]'
        )
    return imp, oob, oos
