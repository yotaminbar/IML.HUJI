import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from IMLearn.metrics.loss_functions import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = [], [], None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.D_ = np.repeat(1/len(y), len(y))  # uniform dist

        joint = [X, y]

        for i in range(self.iterations_):
            if i == 0:
                bootstrap = joint
            else:
                num_to_resample = len(self.D_)
                idx = np.random.choice(num_to_resample, size=num_to_resample, p=self.D_, replace=True)
                bootstrap = [joint[0][idx], joint[1][idx]]

            # go.Figure(data=[go.Scatter(x=bootstrap[0][:, 0], y=bootstrap[0][:, 1], mode="markers", showlegend=False,
            #                            marker=dict(color=bootstrap[1], line=dict(color="black", width=1),
            #                                        colorscale=[custom[0], custom[-1]]))],).show()

            self.models_.append(self.wl_())

            self.models_[-1].fit(bootstrap[0], bootstrap[1] * self.D_)

            pred = self.models_[-1].predict(bootstrap[0])

            # print(misclassification_error(bootstrap[1], pred))

            # self.weights_.append(1/2 * np.log(len(y) / (misclassification_error(bootstrap[1], pred, False)) - 1))
            epsilon = self.D_ @ (pred != bootstrap[1])
            self.weights_.append(1/2 * np.log((1 / epsilon) - 1))

            # print(self.weights_[-1], misclassification_error(bootstrap[1], pred))

            self.D_ = self.D_ * np.exp(-self.weights_[-1] * bootstrap[1] * pred)
            self.D_ /= self.D_.sum()

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        result = np.repeat(0, X.shape[0])
        for i in range(T):
            result = result + self.weights_[i] * self.models_[i].predict(X)
        return np.sign(result)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
