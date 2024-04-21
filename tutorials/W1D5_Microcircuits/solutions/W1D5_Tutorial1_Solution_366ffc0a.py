windows = np.linspace(1, 91, 10)
ema_values = [ema(sig, int(window)) for window in windows]