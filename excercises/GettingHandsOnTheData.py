# import all libraries to be used in this exercise

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotchecker import LinePlotChecker


def normalize_features(features):
    max = np.max(features)
    min = np.min(features)
    return (features - min) / (max - min)


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

    # Show cryptocurrency prices over transaction date
    plt.figure(2)
    axis = plt.gca()
    plt.plot(bitcoin_date, bitcoin, label=("Bitcoin"))
    plt.plot(ethereum_date, ethereum, label=("Ethereum"))
    plt.title(r'$\bf{Figure\ 2.}$ Normalized cryptocurrency prices')
    plt.xlabel('Date')
    plt.ylabel('Normalized price')
    plt.xticks(rotation=20)
    plt.legend()


# Tests
solution1()
assert len(bitcoin) == 948, "Your bitcoin data is incorrect length"
assert len(ethereum) == 948, "Your ethereum data is incorrect length"
assert np.max(bitcoin) == np.max(ethereum), "Incorrect max values after normalisation"
assert np.min(bitcoin) == np.min(ethereum), "Incorrect min values after normalisation"

# Run test that check that plot renders correctly. Requires plotchecker to be installed.
pc = LinePlotChecker(axis)
pc.assert_num_lines(2)
pc.find_permutation('title', r'$\bf{Figure\ 2.}$ Normalized cryptocurrency prices')
pc.find_permutation('xlabel', 'Date')
pc.find_permutation('ylabel', 'Normalized price')
pc.assert_labels_equal(['Bitcoin', 'Ethereum'])