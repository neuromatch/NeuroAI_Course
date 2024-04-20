num_winds = 10
windows = np.linspace(2,92,num_winds)
rows = frame.shape[0]
cols = frame.shape[1]

ema_values_x = [np.array([ema(frame[row], int(window))  for row in range(rows)]) for window in windows]

# deifne ema_values_y
ema_values_y = [np.array([ema(frame[:, col], int(window))  for col in range(cols)]) for window in windows]

visualize_images_ema(frame, ema_values_x, ema_values_y)