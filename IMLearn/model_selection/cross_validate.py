from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score = 0
    validation_score = 0

    for i in range(cv):
        validation_indices = np.arange((i * len(X) / cv), (i + 1) * len(X) / cv - 1, dtype=np.int16)
        validation_set_X = X[validation_indices]
        validation_set_y = y[validation_indices]
        training_X = np.delete(X, validation_indices)
        training_y = np.delete(y, validation_indices)

        estimator.fit(training_X, training_y)

        train_score += estimator.loss(training_X, training_y)
        validation_score += scoring(validation_set_y, estimator.predict(validation_set_X))

    return train_score / cv, validation_score / cv





