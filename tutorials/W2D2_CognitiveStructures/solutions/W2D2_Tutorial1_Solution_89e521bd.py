
def loss(y_true, y_pred):
    """Calculate RMSE loss between true and predicted values (note, that loss is not normalized by the shape).

    Inputs:
    - y_true (np.ndarray): true values.
    - y_pred (np.ndarray): predicted values.

    Outputs:
    - loss (float): loss value.
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

def test_performance(xs, ys, test_sizes):
    """Fit linear regression to the provided data and evaluate the performance with RMSE loss for different test sizes.

    Inputs:
    - xs (np.ndarray): input data.
    - ys (np.ndarray): output data.
    - test_size (list): list of the test sizes.
    """
    performance = []

    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(xs, ys, random_state=1, test_size=test_size)
        regr = LinearRegression().fit(X_train, y_train)
        performance.append(np.copy(loss(y_test, regr.predict(X_test))))
    return performance