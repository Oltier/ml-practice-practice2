import numpy as np
import pandas as pd


def normalize(x, min, max):
    return (x - min) / (max - min)


def normalize_features(features):
    return np.array(list(map(lambda i: normalize(i, np.min(features), np.max(features)), features)))

global df_bc, df_eth, bitcoin, ethereum, axis, best_alpha
# read in historic Bitcoin and Ethereum statistics data from the files "BTC-USD.csv" and "ETH-USD.csv"

df_bc = pd.read_csv("BTC-USD.csv", parse_dates=['Date'])
df_eth = pd.read_csv("ETH-USD.csv", parse_dates=['Date'])

# read the dates for the individual records from column "Date"
bitcoin_date = df_bc.Date.values
ethereum_date = df_eth.Date.values

xs = np.array(df_bc.Close.values)
ys = np.array(df_eth.Close.values)

bitcoin = normalize_features(xs)
ethereum = normalize_features(ys)

global x, y
x = bitcoin
y = ethereum

# plt.show()