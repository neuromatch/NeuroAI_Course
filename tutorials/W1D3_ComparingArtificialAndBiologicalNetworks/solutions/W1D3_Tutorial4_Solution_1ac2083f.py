
stim_idx = [0,1] # change stimulus index to visualize another pair of stimuli
m_dims_list = np.power(2, np.arange(1,10))
true_dist, projected_dist = {}, {}
for i, n_neurons in enumerate(n_neurons_list):
    data = clean_dataset[n_neurons].sel({"stim": stim_idx})
    # Let's first recalculate the ground truth euclidean rdm again, without normalization by the number of neurons this time.
    true_dist[n_neurons] = calc_rdm(data, method='euclidean', noise=None, normalize_by_channels=False).dissimilarities.item()

    projected_dist[n_neurons]=[]
    for m_dims in m_dims_list:
        A = np.random.normal(loc=0, scale=1, size=(n_neurons, m_dims))
        A *= np.sqrt(1/m_dims)
        transformed_data = (data.values @ A)
        transformed_data = np2xr(transformed_data, coords={'stim': data.stim.values, 'neuron': np.arange(m_dims)})
        rdm = calc_rdm(transformed_data, method='euclidean', noise=None, normalize_by_channels=False)
        projected_dist[n_neurons].append(rdm.dissimilarities.item())
    projected_dist[n_neurons] = np.array(projected_dist[n_neurons])

plot_distance_after_projection(true_dist, projected_dist, n_neurons_list, m_dims_list)