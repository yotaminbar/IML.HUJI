import os.path

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def date_mapper(date: float):
    """
    map all dates from 20140101 to increasing naturals every
    month
    """
    date /= 100
    month = int(date) - int(date / 100) * 100
    date /= 100
    year = int(date) - 2014
    return year * 12 + month


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house_prices = pd.read_csv(filename)
    # drop ID, lat, long
    house_prices.drop(labels=["id", "lat", "long"], axis=1, inplace=True)

    house_prices.dropna(inplace=True)

    # changing selling date to increasing naturals starting 2014
    # know this may be a problem during scaling to modern use, but i'm interested to see if price increases with month
    # ordinal data
    house_prices.replace(to_replace="T000000", value="", regex=True, inplace=True)
    house_prices['date'] = pd.to_numeric(house_prices['date'])
    house_prices.dropna(subset=['date'], inplace=True)  # drop null dates
    house_prices['date'] = house_prices['date'].apply(date_mapper)

    # drop prices less than 1000
    house_prices.drop(house_prices[house_prices.price < 1000].index, inplace=True)

    # drop bedrooms less than less than 1
    house_prices.drop(house_prices[house_prices.bedrooms < 1].index, inplace=True)

    # drop non positive bathrooms
    house_prices.drop(house_prices[house_prices.bathrooms <= 0].index, inplace=True)

    # drop non positive bathrooms, sqft_living, sqft_lot,waterfront,view,condition,grade,sqft_above,
    # sqft_basement, sqft_living15,sqft_lot15
    house_prices.drop(house_prices[house_prices.bathrooms <= 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.sqft_living <= 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.sqft_lot <= 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.waterfront < 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.waterfront > 1].index, inplace=True)
    house_prices.drop(house_prices[house_prices.view < 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.condition < 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.grade < 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.sqft_above < 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.sqft_basement < 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.sqft_living15 <= 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.sqft_lot15 <= 0].index, inplace=True)
    house_prices.drop(house_prices[house_prices.yr_built < 1492].index, inplace=True)
    house_prices.drop(house_prices[house_prices.yr_built > 2022].index, inplace=True)
    house_prices.drop(house_prices[house_prices.yr_renovated > 2022].index, inplace=True)

    # drop non relevant zip codes:
    house_prices.drop(house_prices[house_prices.zipcode < 98000].index, inplace=True)
    house_prices.drop(house_prices[house_prices.sqft_lot15 > 98999].index, inplace=True)

    # split zip code to one hot
    # house_prices.zipcode = pd.DataFrame({'zipcode': list(str(set(house_prices.zipcode.tolist())))})
    # house_prices = pd.get_dummies(house_prices)
    one_hot = pd.get_dummies(house_prices['zipcode'])
    house_prices.drop('zipcode', axis=1, inplace=True)
    house_prices = house_prices.join(one_hot)

    # not sure this is ok, but I attempt to make the renovated data more linear:
    # instead of renovated 0 or year -> replace with years since construction / renovation & renovated yes or no
    # is_renov = house_prices.yr_renovated.apply(lambda x: min(x, 1))
    # y_cons_renov = house_prices.date / 12 + 2014 - house_prices[['yr_built', 'yr_renovated']].max(axis=1)
    # is_renov.rename('is_renov', inplace=True)
    # y_cons_renov.rename('y_cons_renov', inplace=True)
    #
    # # remove column yr_renovated and add the two above:
    # house_prices.drop('yr_renovated', axis=1, inplace=True)
    # house_prices = house_prices.join(is_renov)
    # house_prices = house_prices.join(y_cons_renov)

    # print(house_prices.iloc[0])
    # print(house_prices.shape[0])

    # split prices:
    prices = house_prices.price
    house_prices.drop('price', axis=1, inplace=True)

    return house_prices, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for i in range(X.shape[1]):
        cov_mat = np.cov(X.iloc[:, i], y)
        pearson = cov_mat[0][1] / np.sqrt(np.prod(np.diag(cov_mat)))
        fig = go.Figure([go.Scatter(x=X.iloc[:, i], y=y, mode="markers", marker=dict(color="red"))],
                        layout=go.Layout(title=r"$\text{Feature: " + str(X.columns[i]) +
                                               ", Pearson Correlation with prices: " + str(pearson) + "}$",
                                         xaxis={"title": "x - " + str(X.columns[i])},
                                         yaxis={"title": "y - price"},
                                         height=400))
        fig.write_image(output_path + "/" + str(X.columns[i]) + ".png")
        # fig.show()



if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    data = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(data[0], data[1])


    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(data[0], data[1], train_proportion=.75)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # X_tst = X_test.to_numpy()
    # y_tst = y_test.to_numpy()
    #
    # joint = X_train.join(y_train)
    # p_vals = np.linspace(0.1, 1, 91)
    # reg = LinearRegression()
    # mean_loss_p = []
    # std = []
    # ci_plus = []  # confidence interval
    # ci_minus = []  # confidence interval
    #
    # for p in p_vals:
    #     loss_p = []
    #     for i in range(10):
    #         sample = pd.DataFrame.sample(joint, frac=p)
    #         y_smpl = sample.price.to_numpy()
    #         X_smpl = sample.drop('price', axis=1).to_numpy()
    #         reg.fit(X_smpl, y_smpl)
    #         loss_p.append(reg.loss(X_tst, y_tst))
    #
    #     mean_loss_p.append(np.mean(loss_p))
    #     std.append(np.std(loss_p))
    #
    # fig = go.Figure([go.Scatter(x=p_vals, y=mean_loss_p, mode="lines", marker=dict(color="blue"), name="loss"),
    #                  go.Scatter(x=p_vals, y=np.array(mean_loss_p) + 2*np.array(std), fill=None, mode="lines",
    #                             line=dict(color="lightgrey"),
    #                             name=r"$\pm 2 \sigma$"),
    #                  go.Scatter(x=p_vals, y=np.array(mean_loss_p) - 2*np.array(std), fill='tonexty', mode="lines",
    #                             line=dict(color="lightgrey"),
    #                             showlegend=False)],
    #                 layout=go.Layout(title=r"$\text{Loss as Function of Train Sample Size}$",
    #                                  xaxis={"title": "x - Percent of Train data"},
    #                                  yaxis={"title": "y - Loss"},
    #                                  height=400))
    #
    # fig.show()

    np_train = X_train.to_numpy()
    np_train_res = y_train.to_numpy()
    np_test = X_test.to_numpy()
    np_test_res = y_test.to_numpy()
    reg = LinearRegression()
    reg.fit(np_train, np_train_res)

    # compute R^2
    print(1 - ((np_test_res - reg.predict(np_test)) @ (np_test_res - reg.predict(np_test))) /
          ((np_test_res - np.mean(np_test_res)) @ (np_test_res - np.mean(np_test_res))))


    """
    notes from forum:
    zipcode handling
    utils split does so not randomly
    """