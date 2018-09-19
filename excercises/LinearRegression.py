from functools import reduce
from matplotlib import pyplot as plt

from excercises.Globals import *


def feature_matrix(x: np.ndarray):
    # Generate x_i = (x_i, 1) feature matrix
    # x_i = ...
    # YOUR CODE HERE
    x_i = np.array(list(map(lambda i: [i, 1], x)))
    return x_i


def fit(X: np.ndarray, y: np.ndarray):
    # STUDENT TASK ###
    # Compute optimal w by replacing '...' with your solution.
    # Hints: Check out numpy's linalg.inv(), dot() and transpose() functions.
    # w_opt = ...
    # YOUR CODE HERE
    w_opt = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    return w_opt


# Predict new y data
# return numpy.ndarray:
#      [ [y_pred_1]
#          ....
#        [y_pred_N]]
def predict(X: np.array, w_opt: np.array):
    ### STUDENT TASK ###
    ## Predict new y data by replacing '...' with your solution.
    ## Hint! Use X and w_opt to get necessary matrices.
    # y_pred ...
    # YOUR CODE HERE
    y_pred = np.array(list(map(lambda x: w_opt.T.dot(x), X)))
    return y_pred


# Calculate empirical error of the prediction
# return float
def empirical_risk(X: np.array, y: np.array, w_opt: np.array):
    ### STUDENT TASK ###
    ## Compute empirical error by replacing '...' with your solution.
    ## Hints! Use X, Y and w_opt to get necessary matrices.
    ##        Check out numpy's dot(), mean(), power() and subtract() functions.
    # empirical_error = ...
    # YOUR CODE HERE
    N = len(X)
    w_opt_t = w_opt.T
    empirical_error = np.multiply((1 / N),reduce(lambda acc, curr: squared_error_loss(curr[0], curr[1], w_opt_t) + acc,
                                       zip(X, y), 0))
    return empirical_error


def squared_error_loss(x_i, y_i, w_opt_t):
    return np.power(np.subtract(y_i, np.dot(w_opt_t, x_i)), 2)


def linearRegression(X, y):
    ### STUDENT TASK ###
    ## Calculate X, Y, w_opt and empirical_error
    ## Hints! Use featureMatrix() and labelVector() to get necessary matrices, X and Y.
    # X = ...
    # Y = ...
    # w_opt=...
    # empirical_error=...
    # YOUR CODE HERE
    X = feature_matrix(X)
    Y = label_vector(y)
    w_opt = fit(X, Y)
    empirical_error = empirical_risk(X, Y, w_opt)
    return w_opt, empirical_error


def label_vector(y: np.ndarray):
    # Reshape y to ensure correct behavior when doing matrix operations
    return np.reshape(y, (len(y), 1))


def draw_plot(x, y, title=''):
    w_opt, empirical_error = linearRegression(x, y)
    x_pred = np.linspace(0, 1.1, 100)
    y_pred = predict(feature_matrix(x_pred), w_opt)
    # Plot data points and linear regression fitting line
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    plt.plot(x_pred, y_pred, 'r', label=("Empirical = %.4f" % empirical_error))
    plt.title(title)
    plt.xlabel('Bitcoin')
    plt.ylabel('Ethereum')
    plt.legend()
    global axis
    axis = plt.gca()
    # plt.show()


######### Linear regression model for x and y data #########
draw_plot(x, y, r'$\bf{Figure\ 4.}$ Normalized cryptocurrency prices')

w_opt, empirical_error = linearRegression(x, y)
assert empirical_error < 0.015
w_opt, empirical_error = linearRegression([0, 1, 2, 3], [0, 1, 2, 3])
# Because of computational rounding errors, empirical error is almost never exactly 0
assert empirical_error < 1e-30
