T_ar = np.arange(len(sig))

# create a set of 20 signals with different frequencies, from 1 to 101, using sin(2\pi f t).
freqs = np.linspace(0.001,1, 100)
set_sigs = [np.sin(f*T_ar) for f in freqs]

dictionary = np.vstack(set_sigs).T