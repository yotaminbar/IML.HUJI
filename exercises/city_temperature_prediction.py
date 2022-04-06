import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

from datetime import datetime


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    daily_temp_df = pd.read_csv(filename, parse_dates=['Date'])
    countries = ['Jordan', 'Israel', 'South Africa', 'The Netherlands']
    cities = ['Amman', 'Tel Aviv', 'Amsterdam', 'Capetown']

    daily_temp_df.dropna(inplace=True)

    # daily_temp_df = daily_temp_df[daily_temp_df.Country.isin(countries)]
    # daily_temp_df = daily_temp_df[daily_temp_df.City.isin(cities)]

    # drop less than -30 (ams), more that 60 (isr)
    daily_temp_df.drop(daily_temp_df[daily_temp_df.Temp < -30].index, inplace=True)
    daily_temp_df.drop(daily_temp_df[daily_temp_df.Temp > 60].index, inplace=True)

    # drop time greater than now
    daily_temp_df.drop(daily_temp_df[daily_temp_df.Date > datetime.now()].index, inplace=True)

    # create day of year
    d_year = daily_temp_df.Date.apply(lambda x: x.timetuple().tm_yday)
    d_year.rename('DayOfYear', inplace=True)

    # add day_year
    daily_temp_df = daily_temp_df.join(d_year)

    return daily_temp_df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    daily_temp = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    isr_temp = daily_temp.drop(daily_temp.Country[daily_temp.Country != 'Israel'].index)

    # plot temp as function of year day
    isr_temp.Year = isr_temp.Year.astype(str)
    fig1 = px.scatter(isr_temp, x='DayOfYear', y='Temp', color='Year',
                      title='Temperature in Tel Aviv as Function of Day in Year')
    fig1.show()

    # group by month, show std
    fig2 = px.bar(isr_temp.groupby('Month').Temp.agg(np.std),
                  title='SD of daily temperature per Month in Tel-Aviv')
    fig2.update_layout(
        xaxis_title="Month",
        yaxis_title="SD of daily temp",
        showlegend=False
    )
    fig2.show()


    # Question 3 - Exploring differences between countries
    que3 = daily_temp.groupby(['Country', 'Month']).Temp.agg([np.mean, np.std]).reset_index()
    fig3 = px.line(que3, x='Month', y='mean', color='Country', error_y='std')
    fig3.update_layout(
        yaxis_title="Mean Temp",
        title="Average Temp per Month for Each Country + SD"
    )
    fig3.show()


    # Question 4 - Fitting model for different values of `k`
    X_train, y_train, X_test, y_test = split_train_test(isr_temp.DayOfYear, isr_temp.Temp, 0.75)
    X_train = X_train.squeeze()
    X_test = X_test.squeeze()
    k_loss = []

    for k in range(1, 11):
        poltFit = PolynomialFitting(k)
        poltFit.fit(X_train, y_train)
        k_loss.append(np.round(poltFit.loss(X_test, y_test), decimals=2))
        print("Loss for " + str(k) + ": " + str(k_loss[-1]))

    px.bar(x=np.linspace(1,10,10), y=k_loss).update_layout(
        yaxis_title="MSE Loss",
        xaxis_title="Polynomial deg k",
        title="Polyfit Loss as function of K"
    ).show()


    # Question 5 - Evaluating fitted model on different countries
    # get best k (several minima taken into account):
    best_k = np.argmin(k_loss) + 1
    poltFitQ5 = PolynomialFitting(best_k)
    poltFitQ5.fit(isr_temp.DayOfYear, isr_temp.Temp)
    x = list(set(daily_temp.Country))
    x.remove('Israel')
    y = []
    for place in x:
        data = daily_temp.drop(daily_temp.Country[daily_temp.Country != place].index)
        X_data = data.DayOfYear
        y_data = data.Temp
        y.append(poltFitQ5.loss(X_data, y_data))

    px.bar(x=x, y=y, title="Prediction Loss from Model Fitted on Israeli Data, k= " + str(best_k),
           color=x).update_layout(
        yaxis_title="MSE Loss",
        xaxis_title="Country",
    ).show()




