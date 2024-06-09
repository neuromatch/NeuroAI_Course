num_taus = 10

# create taus
taus = np.linspace(1, 91, num_taus).astype(int)

# create taus_list
taus_list = [np.abs(sig[tau:] - sig[:-tau]) for tau in taus]