from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, size=1000)
    univariate = UnivariateGaussian()
    univariate.fit(X)
    print("(" + str(univariate.mu_) + ", " + str(univariate.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    mu = 10
    abs_distance = []
    for m in ms:
        sampleSubset = X[:m]
        univariate.fit(sampleSubset)
        abs_distance.append(np.abs(mu - univariate.mu_))

    go.Figure([go.Scatter(x=ms, y=abs_distance, mode='lines', name=r'$|\mu - \widehat\mu|$'),
               go.Scatter(x=ms, y=[0]*len(ms), mode='lines', name=r'$0$')],
              layout=go.Layout(title=r"$\text{Absolute Distance between Estimator and Expectation"
                                     r" as a Function of Sample Size}$",
                               xaxis_title="$\\text{Number of Samples}$",
                               yaxis_title="$\\text{Absolute distance}$",
                               height=400)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    univariate.fit(X)
    go.Figure([go.Scatter(x=X, y=univariate.pdf(X), mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{PDF of 1000 Drawn Samples}$",
                               xaxis_title="$\\text{Sampled Value}$",
                               yaxis_title="$\\text{Probability Density of Value}$",
                               height=400)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)
    multivariate = MultivariateGaussian()
    multivariate.fit(X)
    print(multivariate.mu_)
    print(multivariate.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)

    opacity = []
    max_log_like = -np.inf

    for val1 in f1:
        for val3 in f3:
            log_like = multivariate.log_likelihood(np.array([val1, 0, val3, 0]), cov, X)
            if log_like > max_log_like:
                max_log_like = log_like
                max_f1 = val1
                max_f3 = val3
            opacity.append(log_like)

    z1 = np.array(opacity).reshape(200, 200)

    go.Figure(go.Heatmap(x=f1, y=f3, z=z1), layout=go.Layout(title="Heatmap", height=800, width=800,
              yaxis_title="f_1 value",
              xaxis_title="f_3 value")).show()

    # Question 6 - Maximum likelihood
    print("max log-likelihood is: " + str(max_log_like))

    print("at f1: " + str(np.round(max_f1, decimals=3)) + " and f3: " + str(np.round(max_f3, decimals=3)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
