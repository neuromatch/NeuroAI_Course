
phi_shifted = phis[200,:][None,:] * encoder.encode([[np.pi/2]])
sims = np.einsum('d,md->m',phi_shifted.flatten(),phis)