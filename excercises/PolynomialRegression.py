from functools import reduce

from matplotlib import pyplot as plt

from excercises.Globals import *
# Linear regression model for feature mapping.
from excercises.LinearRegression import label_vector, fit, predict


def polynomialRegression(x: np.ndarray, y: np.ndarray, degree=1):
    ### STUDENT TASK ###
    ## Calculate w_opt, empirical_error, X and Y
    # X = ...
    # Y = ...
    # w_opt=...
    # empirical_error=...
    # YOUR CODE HERE
    # TODO 1. Calculate for each element of X, the h^(w)(x) (I get another matrix H of the same size)
    # TODO 2. Iterate through each element of X, Y, and H and do the equation. (H already has the values for the average squared error loss function)
    # TODO 3. Find the minimum value of this list.
    X = feature_mapping(x, degree)
    Y = label_vector(y)
    w_opt = fit(X, Y).flatten()
    empirical_error = empirical_risk(X, Y, w_opt, degree)
    return w_opt, empirical_error


# Calculate empirical error of the prediction
# return float
# TODO Empirical risk is currently calculated in a bad way. It should somehow be calculated on an array and minimized
def empirical_risk(X: np.array, y: np.array, w_opt: np.array, degree):
    ### STUDENT TASK ###
    ## Compute empirical error by replacing '...' with your solution.
    ## Hints! Use X, Y and w_opt to get necessary matrices.
    ##        Check out numpy's dot(), mean(), power() and subtract() functions.
    # empirical_error = ...
    # YOUR CODE HERE
    N = len(X)
    M = len(X[0])
    w_opt_t = w_opt.T
    H_poly = np.zeros((N, N))
    for i in range(N):
        for j in range(M):
            H_poly[i][j] = h(w_opt_t, X[i][j], degree)

    average_loss = np.array(list(map(lambda h_i: np.multiply((1 / N), averaged_square_error_loss(h_i, y)), H_poly)))

    empirical_error = np.min(average_loss)
    return empirical_error


def averaged_square_error_loss(h_i, y):
    return reduce(lambda acc, curr: np.add(squared_error_loss(curr[0], curr[1]), acc), zip(h_i, y), 0)


def squared_error_loss(h_i, y_i):
    return np.power(np.subtract(y_i, h_i), 2)


def h(w: np.ndarray, x_i, degree):
    ind = np.arange(degree)
    return reduce(lambda acc, curr: np.add(np.multiply(curr[0], np.power(x_i, curr[1])), acc), zip(w, ind), 0)


# Extract feature to higher dimensional by computing feature mapping
# return numpy.ndarray with following mappings:
#      [[x_1^(d), x_1^(d-1), x_1^(d-2), ... , x_1^(0)]
#        ...         ...        ...     ...     ...
#       [x_N^(d), x_N^(d-1), x_N^(d-2), ... ,x_N^(0)]]
def feature_mapping(x: np.ndarray, degree=1):
    ### STUDENT TASK ####
    ## Compute specified feature mapping by replacing '...' with your solution.
    ## Hints! Use x to get all the feature vectors of the data set.
    ##        Check out numpy's vstack(), hstack() and column_stack() functions.
    # polynomial_features = ...
    # YOUR CODE HERE
    polynomial_features = np.array(list(map(lambda x_i: create_feature_map(x_i, degree), x)))
    return polynomial_features


def create_feature_map(x_i, degree=1):
    return np.array([pow(x_i, i) for i in range(degree, -1, -1)]).T


def draw_plot(x, y, title='', degree=1):
    w_opt, empirical_error = polynomialRegression(x, y, degree)

    # Change feature values into continues one from 0 to 1.
    x_pred = np.linspace(0, 1.1, 100)

    # predict new y values using feature mapping
    y_pred = predict(feature_mapping(x_pred, degree), w_opt)

    # Plot data points and linear regression fitting line
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    plt.plot(x_pred, y_pred, 'r')
    plt.plot(x_pred, y_pred, 'r', label=("Empirical = %.4f" % empirical_error))
    plt.title(title)
    plt.xlabel('Bitcoin')
    plt.ylabel('Ethereum')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    # plt.show()


######### Linear regression model for x and y data #########
draw_plot(x, y, r'$\bf{Figure 5.}$Normalized cryptocurrency prices', degree=11)

w_opt, empirical_error = polynomialRegression(x, y)
print(empirical_error)
assert empirical_error < 0.01
w_opt, empirical_error = polynomialRegression([0, 1, 2, 3], [0, 1, 2, 3])
# Because of computational rounding errors, empirical error is almost never 0
assert empirical_error < 1e-30
for i in range(0, 100):
    x_test = feature_mapping([0, 1, 2, 3], i)
    assert x_test.shape == (4, 1 + i)
