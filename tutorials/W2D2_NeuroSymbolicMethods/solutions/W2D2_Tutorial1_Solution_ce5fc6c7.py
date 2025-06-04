
phi_shifted = phis[200] * X**-3.1
sims = np.array([spa.dot(phi_shifted, p) for p in phis])