set_seed(42)

n_hid = 2

n_reps = 10 # Number of networks to train

def plot_predictions(n_hid, n_reps):
    """
    Generate train and test data for `n_reps` times, fit it for network with hidden size `n_hid` and plot prediction values.

    Inputs:
    - n_hid (int): size of hidden layer.
    - n_reps (int): number of data regenerations.
    """
    with plt.xkcd():
        plt.plot(x_test, y_test,linewidth=4,label='Test data')
        plt.plot(x_train, y_train,'o',label='Training data')

        train_err, test_err, y_pred = fit_relu(x_train, y_train, x_test, y_test, n_hidden=n_hid)
        plt.plot(x_test, y_pred, color='g', label='Prediction')

        for rep in range(n_reps-1):
            train_err, test_err, y_pred = fit_relu(x_train, y_train, x_test, y_test, n_hidden=n_hid)
            plt.plot(x_test, y_pred, color='g', alpha=.5, label='_')

        plt.legend()
        plt.xlabel('Input Feature')
        plt.ylabel('Target Output')
        plt.title('Number of Hidden Units = {}'.format(n_hid))
        plt.show()

plot_predictions(n_hid, n_reps)