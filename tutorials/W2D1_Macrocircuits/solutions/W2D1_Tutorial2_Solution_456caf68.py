set_seed(42)

n_hids = np.unique(np.round(np.logspace(0, 3, 10))).astype(int)

std_devs = np.linspace(0, 2, 3)

def plot_error(x_train, y_train, x_test, y_test, std_devs, n_hids, n_hidden = 10, n_reps = 100, reg = 0):
    """
    Plot mean test error for distinct values of noise added to train dataset.

    Inputs:
    - x_train (np.ndarray): train input data.
    - y_train (np.ndarray): train target data.
    - x_test (np.ndarray): test input data.
    - y_test (np.ndarray): test target data.
    - std_devs (np.ndarray): different standard deviation values for noise.
    - n_hids (np.ndarray): different values for hidden layer size.
    - n_hidden (int, default = 10): size of hidden layer.
    - n_reps (int, default = 100): number of resamples for data.
    - reg (float, default = 0): regularization constant.
    """
    with plt.xkcd():
        for sd in std_devs:
            test_errs = [sweep_test(x_train, y_train + np.random.normal(0,sd,y_train.shape), x_test, y_test, n_hidden = n_hid, n_reps = n_reps, reg = reg * (1 + sd)) for n_hid in n_hids]
            plt.loglog(n_hids,test_errs,'o-',label="std={}".format(sd))

        plt.legend()
        plt.xlabel('Number of Hidden Units')
        plt.ylabel('Test Error')
        plt.show()

plot_error(x_train, y_train, x_test, y_test, std_devs, n_hids)