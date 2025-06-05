
#features - rules
X_train = np.array([r.v for r in rules]).squeeze()

#output - a* for each rule
y_train = np.array([
    (vocab[ant_names[0]] + vocab['NOT']*vocab[cons_names[0]]).normalized().v,
    (vocab[ant_names[1]] + vocab['NOT']*vocab[cons_names[1]]).normalized().v,
]).squeeze()

regr = MLPRegressor(random_state=1, hidden_layer_sizes=(1024,1024), max_iter=1000).fit(X_train, y_train)

a_mlp = regr.predict(new_rule.v[None,:])

mlp_sims = np.einsum('nd,md->nm', action_space, a_mlp)