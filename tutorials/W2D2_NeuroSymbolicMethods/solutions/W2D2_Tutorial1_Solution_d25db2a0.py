
phi_shifted = phis[200,:][None,:] * new_encoder.encode([[np.pi/2]])
shifted_real_line_sims = phi_shifted.flatten() @ phis.T