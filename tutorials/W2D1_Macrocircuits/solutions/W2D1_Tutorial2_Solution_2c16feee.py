
def fit_relu(x_train, y_train, x_test, y_test, n_hidden = 10, reg = 0):
    """
    Fit second layer of network by solving via linear regression for the given training data and evaluate the performance.

    Inputs:
    - x_train (np.ndarray): train input data.
    - y_train (np.ndarray): train target data.
    - x_test (np.ndarray): test input data.
    - y_test (np.ndarray): test target data.
    - n_hidden (int, default = 10): size of hidden layer.
    - reg (float, default = 0): regularization term.

    Outputs:
    - train_err (float): train error value.
    - test_err (float): test error value.
    - y_pred (np.ndarray): array of predicted values for test data.
    """

    # Define network architecture
    n_inputs = 1  # Number of input features
    n_outputs = 1  # Number of output units

    # Layer 1 (Input -> Hidden)
    W1 = np.random.normal(0, 1, (n_inputs, n_hidden))  # Random weights
    b1 = np.random.uniform(-np.pi, np.pi, size = (1, n_hidden))  # Bias

    # Layer 2 (Hidden -> Output)
    W2 = np.zeros((n_hidden, n_outputs))  # Initialize weights to zero

    # Forward propagation
    def forward_prop(X):
        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = a1.dot(W2)
        return z2

    # Fit second layer weights with linear regression
    hidden = np.maximum(0, x_train.dot(W1) + b1)  # Hidden layer activations
    if reg == 0:
        # Pseudo-inverse solution
        hidden_pinv = np.linalg.pinv(hidden)
        W2 = hidden_pinv.dot(y_train)
    else:
        # We use linalg.solve to find the solution to (H'H + reg*I) * W2 = H'y,
        # equivalent to W2 = (H'H + reg*I)^(-1) * H'y
        W2 = np.linalg.solve(hidden.T @ hidden + reg * np.eye(n_hidden), hidden.T @ y_train)

    # Train Error
    y_pred = forward_prop(x_train)
    train_err = np.mean((y_train-y_pred)**2/2)

    # Test Error
    y_pred = forward_prop(x_test)
    test_err = np.mean((y_test-y_pred)**2/2)

    return train_err, test_err, y_pred