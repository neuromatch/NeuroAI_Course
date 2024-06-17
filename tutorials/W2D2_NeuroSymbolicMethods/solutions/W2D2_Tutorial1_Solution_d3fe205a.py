
phi_shifted = phis[200,:][None,:] * encoder.encode([[np.pi/2]])
sims = phi_shifted.flatten() @ phis.T