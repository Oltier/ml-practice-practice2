# import all libraries to be used in this exercise

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display

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
    #bitcoin = [...]
    #ethereum = [...]

    # YOUR CODE HERE
    raise NotImplementedError()

    # Show cryptocurrency prices over transaction date
    axis = plt.gca()
    plt.plot(bitcoin_date,bitcoin, label=("Bitcoin"))
    plt.plot(ethereum_date,ethereum, label=("Ethereum"))
    plt.title(r'$\bf{Figure\ 2.}$ Normalized cryptocurrency prices')
    plt.xlabel('Date')
    plt.ylabel('Normalized price')
    plt.xticks(rotation=20)
    plt.legend()

# TESTS
# solution1()
# assert len(bitcoin) == 948,"Your bitcoin data is incorrect length"
# assert len(ethereum) == 948, "Your ethereum data is incorrect length"
# assert np.max(bitcoin)== np.max(bitcoin), "Incorrect max values after normalisation"
# assert np.min(bitcoin)== np.min(bitcoin), "Incorrect min values after normalisation"
#
# # Run test that check that plot renders correctly. Requires plotchecker to be installed.
# from plotchecker import LinePlotChecker
# pc = LinePlotChecker(axis)
# pc.assert_num_lines(2)
# pc.find_permutation('title',r'$\bf{Figure\ 2.}$ Normalized cryptocurrency prices')
# pc.find_permutation('xlabel','Date')
# pc.find_permutation('ylabel','Normalized price')
# pc.assert_labels_equal(['Bitcoin','Ethereum'])