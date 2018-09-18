def featureMatrix(x):
    # Generate x_i = (x_i, 1) feature matrix
    # x_i = ...
    # YOUR CODE HERE
    raise NotImplementedError()
    return x_i

def fit(x, y):
    ### STUDENT TASK ###
    ## Compute optimal w by replacing '...' with your solution.
    ## Hints: Check out numpy's linalg.inv(), dot() and transpose() functions.
    # w_opt = ...
    # YOUR CODE HERE
    raise NotImplementedError()
    return w_opt

# Predict new y data
# return numpy.ndarray:
#      [ [y_pred_1]
#          ....
#        [y_pred_N]]
def predict(X, w_opt):
    ### STUDENT TASK ###
    ## Predict new y data by replacing '...' with your solution.
    ## Hint! Use X and w_opt to get necessary matrices.
    # y_pred ...
    # YOUR CODE HERE
    raise NotImplementedError()
    return y_pred

# Calculate empirical error of the prediction
# return float
def empirical_risk(X, Y, w_opt):
    ### STUDENT TASK ###
    ## Compute empirical error by replacing '...' with your solution.
    ## Hints! Use X, Y and w_opt to get necessary matrices.
    ##        Check out numpy's dot(), mean(), power() and subtract() functions.
    # empirical_error = ...
    # YOUR CODE HERE
    raise NotImplementedError()
    return empirical_error

def linearRegression(x, y):
    ### STUDENT TASK ###
    ## Calculate X, Y, w_opt and empirical_error
    ## Hints! Use featureMatrix() and labelVector() to get necessary matrices, X and Y.
    # X = ...
    # Y = ...
    # w_opt=...
    # empirical_error=...
    # YOUR CODE HERE
    raise NotImplementedError()
    return w_opt, empirical_error

def labelVector(y):
    # Reshape y to ensure correct behavior when doing matrix operations
    return np.reshape(y,(len(y),1))

def draw_plot(x, y, title=''):
    w_opt, empirical_error = linearRegression(x, y)
    x_pred = np.linspace(0,1.1,100)
    y_pred = predict(featureMatrix(x_pred), w_opt)
    # Plot data points and linear regression fitting line
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    plt.plot(x_pred,y_pred,'r', label=("Empirical = %.4f" % empirical_error))
    plt.title(title)
    plt.xlabel('Bitcoin')
    plt.ylabel('Ethereum')
    plt.legend()
    axis = plt.gca()


######### Linear regression model for x and y data #########
draw_plot(x, y, r'$\bf{Figure\ 4.}$ Normalized cryptocurrency prices')




w_opt, empirical_error = linearRegression(x, y)
assert empirical_error < 0.015
w_opt, empirical_error = linearRegression([0,1,2,3], [0,1,2,3])
# Because of computational rounding errors, empirical error is almost never exactly 0
assert empirical_error < 1e-30