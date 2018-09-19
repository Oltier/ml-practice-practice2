# import all libraries to be used in this exercise

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotchecker import LinePlotChecker, ScatterPlotChecker


def normalize(x, min, max):
    return (x - min) / (max - min)


def normalize_features(features):
    return np.array(list(map(lambda i: normalize(i, np.min(features), np.max(features)), features)))


def solution1():
    # We assign Global variables for testing purposes
    global df_bc, df_eth, bitcoin, ethereum, axis, best_alpha
    # read in historic Bitcoin and Ethereum statistics data from the files "BTC-USD.csv" and "ETH-USD.csv"

    df_bc = pd.read_csv("BTC-USD.csv", parse_dates=['Date'])
    df_eth = pd.read_csv("ETH-USD.csv", parse_dates=['Date'])

    # read the dates for the individual records from column "Date"
    bitcoin_date = df_bc.Date.values
    ethereum_date = df_eth.Date.values

    ### STUDENT TASK ###
    ## read in closing prices for Bitcoin and Ethereum from the column "Close" and computed the rescaled values
    ## Replace '...' with your solution.
    # bitcoin = [...]
    # ethereum = [...]

    # YOUR CODE HERE
    xs = np.array(df_bc.Close.values)
    ys = np.array(df_eth.Close.values)

    bitcoin = normalize_features(xs)
    ethereum = normalize_features(ys)

    x = bitcoin
    y = ethereum

    plt.figure(3, figsize=(8, 8))
    plt.scatter(x, y)
    plt.title(r'$\bf{Figure\ 3.}$Normalized cryptocurrency prices ($x^{(i)},y^{(i)}$)')
    plt.xlabel('normalized Bitcoin prices')
    plt.ylabel('normalized Ethereum prices')
    plt.annotate('$(x^{(i)},y^{(i)})$', xy=(x[913], y[913]), xytext=(0.1, 0.5),
                 arrowprops=dict(arrowstyle="->", facecolor='black'),
                 )
    axis = plt.gca()
    # plt.show()


# Tests
solution1()
pc = ScatterPlotChecker(axis)
assert len(pc.x_data) == 948
assert len(pc.y_data) == 948