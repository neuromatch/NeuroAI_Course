set_seed(42)

std_dev = .2

noise = np.random.normal(0, std_dev, y_train.shape)

n_hid = 500
n_reps = 10

with plt.xkcd():
    plt.plot(x_test, y_test,linewidth=4,label='Test data')
    plt.plot(x_train, y_train + noise,'o',label='Training data')
    train_err, test_err, y_pred = fit_relu(x_train, y_train + noise, x_test, y_test, n_hidden = n_hid)
    plt.plot(x_test, y_pred, color='g', label='Prediction')
    plt.legend()
    plt.xlabel('Input Feature')
    plt.ylabel('Target Output')
    plt.title('Number of Hidden Units = {}'.format(n_hid))
    plt.show()