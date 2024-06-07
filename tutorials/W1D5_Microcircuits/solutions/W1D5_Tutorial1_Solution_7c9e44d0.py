temporal_diff = np.abs(np.diff(sig))
plot_signal(temporal_diff, title = "", ylabel = "$| pixel_t - pixel_{t-1} | $")