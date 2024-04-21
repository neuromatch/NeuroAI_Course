num_taus = 10

# create taus
taus = np.linspace(1,91,num_taus).astype(int)

# create taus_list
taus_list_x = [np.abs(frame[:,tau:] - frame[:,:-tau]) for tau in taus]
taus_list_y = [np.abs(frame[tau:,:] - frame[:-tau,:]) for tau in taus]

plot_spatial_diff_histogram(taus, taus_list_x, taus_list_y)