set_seed(42)
init_scales = np.linspace(0, 3, 5)

n_hids = np.unique(np.round(np.logspace(0, 3, 10))).astype(int)

with plt.xkcd():
    for sd in init_scales:
        test_errs = [sweep_test_init_scale(x_train, y_train, x_test, y_test, init_scale = sd, n_hidden=n_hid, n_reps=100) for n_hid in n_hids]
        plt.loglog(n_hids,test_errs,'o-',label="Init Scale={}".format(sd))

    plt.legend()
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Test Error')
    plt.show()