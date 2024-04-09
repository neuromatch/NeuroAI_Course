
summer_r_squared = [model.score(summer_days_test_norm, summer_prices_test)]
autumn_r_squared = [model.score(autumn_days_test_norm, autumn_prices_test)]
num_epochs = 10

for _ in range(num_epochs - 1):
    #fit new data for one epoch
    model.partial_fit(autumn_days_train_norm, autumn_prices_train)

    #calculate r-squared values on test sets
    summer_r_squared.append(model.score(summer_days_test_norm, summer_prices_test))
    autumn_r_squared.append(model.score(autumn_days_test_norm, autumn_prices_test))

plot_performance(num_epochs, summer_r_squared, autumn_r_squared)