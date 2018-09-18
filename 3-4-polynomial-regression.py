# Linear regression model for feature mapping.
def polynomialRegression(x, y, degree=1):
    ### STUDENT TASK ###
    ## Calculate w_opt, empirical_error, X and Y
    # X = ...
    # Y = ...
    # w_opt=...
    # empirical_error=...
    # YOUR CODE HERE
    raise NotImplementedError()
    return w_opt, empirical_error

# Extract feature to higher dimensional by computing feature mapping
# return numpy.ndarray with following mappings:
#      [[x_1^(d), x_1^(d-1), x_1^(d-2), ... , x_1^(0)]
#        ...         ...        ...     ...     ...
#       [x_N^(d), x_N^(d-1), x_N^(d-2), ... ,x_N^(0)]]
def feature_mapping(x, degree=1):
    ### STUDENT TASK ####
    ## Compute specified feature mapping by replacing '...' with your solution.
    ## Hints! Use x to get all the feature vectors of the data set.
    ##        Check out numpy's vstack(), hstack() and column_stack() functions.
    # polynomial_features = ...
    # YOUR CODE HERE
    raise NotImplementedError()
    return polynomial_features

def draw_plot(x, y, title='', degree=1):
    w_opt, empirical_error = polynomialRegression(x, y, degree)

    # Change feature values into continues one from 0 to 1.
    x_pred = np.linspace(0,1.1,100)

    # predict new y values using feature mapping
    y_pred = predict(feature_mapping(x_pred, degree), w_opt)

    # Plot data points and linear regression fitting line
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    plt.plot(x_pred,y_pred,'r', label=("Empirical = %.4f" % empirical_error))
    plt.title(title)
    plt.xlabel('Bitcoin')
    plt.ylabel('Ethereum')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()
    plt.show()


######### Linear regression model for x and y data #########
draw_plot(x,y,r'$\bf{Figure 5.}$Normalized cryptocurrency prices',degree=11)





w_opt, empirical_error = polynomialRegression(x, y)
assert empirical_error < 0.01
w_opt, empirical_error = polynomialRegression([0,1,2,3], [0,1,2,3])
# Because of computational rounding errors, empirical error is almost never 0
assert empirical_error < 1e-30
for i in range(0,100):
    x_test = feature_mapping([0,1,2,3],i)
    assert x_test.shape == (4,1+i)