
#features - rules
X_train = np.array(rules).squeeze()

#output - a* for each rule
y_train = np.array([
    (vocab[ant_names[0]] + vocab['not']*vocab[cons_names[0]]).normalize(),
    (vocab[ant_names[1]] + vocab['not']*vocab[cons_names[1]]).normalize(),
]).squeeze()

regr = MLPRegressor(random_state=1, hidden_layer_sizes=(1024,1024), max_iter=1000).fit(X_train, y_train)

a_mlp = regr.predict(new_rule)

mlp_sims = action_space @ a_mlp.T