set_seed(42)

n_hids = np.unique(np.round(np.logspace(0, 3, 20))).astype(int)

def sweep_test(x_train, y_train, x_test, y_test, n_hidden = 10, n_reps = 100, reg = 0.0):
    """
    Calculate the mean test error for fitting the second layer of the network for a defined number of repetitions.
    Notice that `init_scale` is always set to 0 in this case.
    Inputs:
    - x_train (np.ndarray): train input data.
    - y_train (np.ndarray): train target data.
    - x_test (np.ndarray): test input data.
    - y_test (np.ndarray): test target data.
    - n_hidden (int, default = 10): size of hidden layer.
    - n_reps (int, default = 100): number of resamples for data.
    - reg (float, default = 0): regularization constant.

    Outputs:
    - (float): mean error for train data.
    """
    return np.mean(np.array([fit_relu(x_train, y_train, x_test, y_test, n_hidden=n_hidden, reg = reg)[1] for _ in range(n_reps)]))

test_errs = [sweep_test(x_train, y_train, x_test, y_test, n_hidden=n_hid, n_reps=100, reg = 0.0) for n_hid in n_hids]

with plt.xkcd():
    plt.loglog(n_hids,test_errs,'o-',label='Test')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Test Error')
    plt.show()