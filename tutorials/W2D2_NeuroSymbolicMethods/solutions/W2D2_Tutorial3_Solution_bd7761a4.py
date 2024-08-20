
def rastrigin(x):
    """Compute Rastrigin function for given array of d-dimenstional vectors.

    Inputs:
    - x (np.ndarray of shape (n, d)): n d-dimensional vectors.

    Outputs:
    - y (np.ndarray of shape (n, 1)): Rastrigin function value for each of the vectors.
    """
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2*np.pi*x), axis=1)

# this code creates 10000 2-dimensional vectors which are going to be served as input to the function (thus, output is of shape (10000, 1))
x0_rastrigin = np.linspace(-5.12, 5.12, 100)
X_rastrigin, Y_rastrigin = np.meshgrid(x0_rastrigin,x0_rastrigin)
xs_rastrigin = np.vstack((X_rastrigin.flatten(), Y_rastrigin.flatten())).T

ys_rastrigin = rastrigin(xs_rastrigin)

plot_3d_function([X_rastrigin],[Y_rastrigin], [ys_rastrigin.reshape(X_rastrigin.shape)], ['Rastrigin Function'])