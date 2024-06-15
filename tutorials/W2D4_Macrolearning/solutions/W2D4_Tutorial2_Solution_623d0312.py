
# Initial r-squared calculations
summer_r_squared = [base_model.score(summer_days_test_norm, summer_prices_test)]
autumn_r_squared = [base_model.score(autumn_days_test_norm, autumn_prices_test)]
num_epochs = 10

# Progress bar integration with tqdm
for _ in tqdm(range(num_epochs - 1), desc="Training Progress"):
    # Fit new data for one epoch
    base_model.partial_fit(autumn_days_train_norm, autumn_prices_train)

    # Calculate r-squared values on test sets
    summer_r_squared.append(base_model.score(summer_days_test_norm, summer_prices_test))
    autumn_r_squared.append(base_model.score(autumn_days_test_norm, autumn_prices_test))

# Final model and plot
model = base_model
plot_performance(num_epochs, summer_r_squared, autumn_r_squared)