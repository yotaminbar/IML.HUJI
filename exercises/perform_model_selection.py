from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.random.uniform(low=-1.2, high=2, size=n_samples)
    y = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    eps = np.random.normal(loc=0, scale=np.sqrt(noise), size=n_samples)

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(x), y + eps, 2 / 3)

    make_subplots(rows=1, cols=1) \
        .add_traces(
        [go.Scatter(x=x, y=y, mode='markers', marker=dict(color="red"), name="true model"),
         go.Scatter(x=train_x[0][:], y=train_y, mode='markers', marker=dict(color="blue"),
                    name="train"),
         go.Scatter(x=test_x[0][:], y=test_y, mode='markers', marker=dict(color="green"),
                    name="test")]
        ,
        rows=1, cols=1) \
        .update_layout(title_text=r"$\text{Question 1 generating data}$", height=300) \
        .show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    data = pd.DataFrame(columns=['k', 'avg_train_error', 'avg_validation_error'])

    for i in range(11):
        train_score, validation_score = \
            cross_validate(PolynomialFitting(i), train_x[0][:].to_numpy(), train_y[:].to_numpy(), mean_square_error, 5)
        data = pd.concat([data, pd.DataFrame({'k': [i], 'avg_train_error': [train_score], 'avg_validation_error': \
            [validation_score]})], ignore_index=True, axis=0)

    fig = go.Figure(
        data=[
            go.Bar(name='avg_train_error', x=data['k'], y=data['avg_train_error'], yaxis='y', offsetgroup=1),
            go.Bar(name='avg_validation_error', x=data['k'], y=data['avg_validation_error'], yaxis='y2', offsetgroup=2)
        ],
        layout={
            'yaxis': {'title': 'avg_train_error'},
            'yaxis2': {'title': 'avg_validation_error', 'overlaying': 'y', 'side': 'right'},
            'xaxis': {'title': 'K'}
        }
    )

    # Change the bar mode
    fig.update_layout(barmode='group', title_text=r"$\text{Question 2}$")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    pf = PolynomialFitting(5)
    pf.fit(train_x[0][:].to_numpy(), train_y[:].to_numpy())

    print(pf.loss(test_x[0][:].to_numpy(), test_y[:].to_numpy()))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)

    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(noise=10, n_samples=1500)
