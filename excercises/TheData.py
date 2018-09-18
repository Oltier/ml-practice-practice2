import pandas as pd  # import Pandas library (and defining shorthand "pd") for reading and manipulating the data files
from IPython.display import display, HTML
from matplotlib import \
    pyplot as plt  # import and define shorthand "plt" for library "pyplot" providing plotting functions

# read in data from csv files
# parse_dates function is used on Date-column to change them from string to date-object

df_bc = pd.read_csv("BTC-USD.csv", parse_dates=['Date'])
df_eth = pd.read_csv("ETH-USD.csv", parse_dates=['Date'])

# Show top rows of each file.
# function "display()" is Jupyter Notebook command to show multiple function outputs from one cell.
display(HTML(df_bc.head(5).to_html(max_rows=5)))
display(HTML(df_eth.head(5).to_html(max_rows=5)))

# Plot original data to same figure
# Example of use: plt.plot(x,y)
plt.plot(df_bc.Date.values, df_bc.Close.values, label=("Bitcoin price (in USD)"))
plt.plot(df_eth.Date.values, df_eth.Close.values, label=("Ethereum price (in USD)"))

# set up figure title and labels for x- and y-axis
plt.title(r'$\bf{Figure\ 1.}$ Bitcoin vs Ethereum')
plt.xlabel('date')
plt.ylabel('price')

# rotate x-ticks by 20 degrees
plt.xticks(rotation=20)

# enable legend-box for plot labels
plt.legend()

# show the plot
plt.show()
