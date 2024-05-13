
summer_r_squared = [base_model.score(summer_days_test_norm, summer_prices_test)]
autumn_r_squared = [base_model.score(autumn_days_test_norm, autumn_prices_test)]
num_epochs = 10

for _ in range(num_epochs - 1):
    #fit new data for one epoch
    base_model.partial_fit(autumn_days_train_norm, autumn_prices_train)

    #calculate r-squared values on test sets
    summer_r_squared.append(base_model.score(summer_days_test_norm, summer_prices_test))
    autumn_r_squared.append(base_model.score(autumn_days_test_norm, autumn_prices_test))

model = base_model
plot_performance(num_epochs, summer_r_squared, autumn_r_squared)