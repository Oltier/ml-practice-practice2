# Calculate empirical error of the prediction
def empirical_risk(X, Y, w_opt):
    # Assign current weight vector according to iterative weight vector
    ### STUDENT TASK ###
    ## Compute empirical error by replacing '...' with your solution.
    ## Hints! Use X, Y and w_opt to get necessary matrices.
    ##        Check out numpy's dot(), mean(), power() and subtract() functions.
    # empirical_error = ...
    # YOUR CODE HERE
    raise NotImplementedError()
    return empirical_error

# Compute gradient
# return numpy.ndarray:
#      [ [w_1]
#          ....
#        [w_d]]
def gradient(X,Y,w):
    N = X.shape[0]
    ### STUDENT TASK ###
    ## Compute gradient by replacing '...' with your solution.
    ## Hint! Use X and Y to get necessary matrices.
    # gradient = ...
    # YOUR CODE HERE
    raise NotImplementedError()
    return gradient

# Run GD for k steps
# a = alpha/learning rate
# k = iteration steps
# returns empirical error array with lenght k: [ Empirical_1, ..., Empirical_k ]
def gradient_descent(x, y, a, k):
    ### STUDENT TASK ###
    ## Hints! Same as in linearRegression()!
    # X = ...
    # Y = ...
    # YOUR CODE HERE
    raise NotImplementedError()
    # Initial weigth vector (all values 0)
    w = np.zeros((X.shape[1], Y.shape[1]))
    empirical_errors = []

    for i in range(k):
        # Calculate gradient
        grad = gradient(X,Y,w)

        ### STUDENT TASK ###
        ## Update weight vector by replacing '...' with your solution.
        # w = ...
        # YOUR CODE HERE
        raise NotImplementedError()
        ### STUDENT TASK ###
        # Calculate Empirical Risk and append the error into empirical_errors array
        # YOUR CODE HERE
        raise NotImplementedError()
    return empirical_errors

def visualize_error(X, y, learning_rates, best_alpha = None):
    fig, axes = plt.subplots(1, 2,figsize=(12, 4))

    for learning_rate in learning_rates:
        # Plot Error against Step Size
        GD_converge=gradient_descent(x,y,learning_rate, 1000)
        axes[0].plot(GD_converge,label=(r'$\alpha=$'+str(learning_rate)))

    axes[0].set_xlabel('Number of Iterations')
    axes[0].set_ylabel('Empirical Error')
    axes[0].legend(loc=0)
    axes[0].set_title(r'$\bf{Figure\ 6.}$Converge of GD')

    for learning_rate in learning_rates:
        # Plot Error against Step Size.
        # Now mark the best converge in red. Use value from best as a correct step size.
        GD_converge=gradient_descent(x,y,learning_rate, 1000)

        if learning_rate == best_alpha:
            axes[1].plot(GD_converge,label=(r'$\alpha=$'+str(learning_rate)), color="red")
        else:
            axes[1].plot(GD_converge,label=(r'$\alpha=$'+str(learning_rate)), color="blue")

    axes[1].set_xlabel('Number of Iterations')
    axes[1].set_ylabel('Empirical Error')
    axes[1].legend(loc=0)
    axes[1].set_title(r'$\bf{Figure\ 7.}$Converge of GD')
    plt.tight_layout()
    return axes, best_alpha

learning_rates = [0.001, 0.01, 0.02, 0.1, 0.2, 0.3, 0.9, 0.98]

### STUDENT TASK ###
# Change best=None into step size from the list that provides the fastest converge. e.g best=1
###
GD_plots, best = visualize_error(x, y, best_alpha=None, learning_rates=learning_rates)



assert best != None, "You haven't specified the best learning rate"
for i in [1,10,100,200,243]:
    res = gradient_descent([0,0.5,1], [0,0.5,1], 1e-5, i)
    assert len(res) == i, "Size of the Error array is incorrect"
    assert np.sum(res)/i < 1, "Your error is way higher than it should be."

# Check that plots are rendered.
from plotchecker import LinePlotChecker
for GD_plot in GD_plots:
    pc = LinePlotChecker(GD_plot)
    pc.assert_num_lines(8)