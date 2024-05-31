n_neurons = 2
stimulus_idx = 0,1 # choose two stimuli
b_j = clean_dataset[n_neurons].loc[stimulus_idx[0]].values # select the stimulus response
b_k = clean_dataset[n_neurons].loc[stimulus_idx[1]].values
# compute the squared euclidean and mahalanobis distance, and then divide the distance by the number of neurons (2)
euclidean_dist = ((b_j-b_k) @ (b_j-b_k).T)/n_neurons
mahalanobis_dist = ((b_j-b_k) @ np.linalg.inv(correlated_cov[n_neurons]) @ (b_j-b_k).T)/n_neurons