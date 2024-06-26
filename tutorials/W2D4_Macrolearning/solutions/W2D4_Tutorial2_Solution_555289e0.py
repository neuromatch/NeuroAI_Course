
def interspersed_training(K: int, num_epochs: int):
    """
    Perform interspersed training for a multi-layer perceptron (MLP) regression model.

    Inputs:
        - K (int): The number of training examples to sample from each season (summer and autumn) in each epoch.
        - num_epochs (int): The number of training epochs.

    Returns:
        - model (MLPRegressor): The trained MLP regression model.
        - summer_r_squared (list): A list containing the R-squared values for the summer season at each epoch.
        - autumn_r_squared (list): A list containing the R-squared values for the autumn season at each epoch.
    """

    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=10000, random_state = 42, solver = "lbfgs")
    summer_r_squared = []
    autumn_r_squared = []

    for _ in tqdm(range(num_epochs), desc="Training Progress"):
        # Sample random training examples from summer and autumn
        sampled_summer_indices = np.random.choice(np.arange(summer_days_train_norm.shape[0]), size=K, replace=False)
        sampled_autumn_indices = np.random.choice(np.arange(autumn_days_train_norm.shape[0]), size=K, replace=False)

        mixed_days_train = np.expand_dims(np.append(autumn_days_train_norm[sampled_autumn_indices], summer_days_train_norm[sampled_summer_indices]), 1)
        mixed_prices_train = np.append(autumn_prices_train[sampled_autumn_indices], summer_prices_train[sampled_summer_indices])
        model.partial_fit(mixed_days_train, mixed_prices_train)

        summer_r_squared.append(model.score(summer_days_test_norm, summer_prices_test))
        autumn_r_squared.append(model.score(autumn_days_test_norm, autumn_prices_test))

    return model, summer_r_squared, autumn_r_squared