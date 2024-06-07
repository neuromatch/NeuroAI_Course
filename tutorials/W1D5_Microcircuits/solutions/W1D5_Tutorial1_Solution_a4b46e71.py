
num_winds = 10
windows = np.linspace(2,92,num_winds)
rows = frame.shape[0]
cols = frame.shape[1]

diff_box_values_x = [np.array([diff_box_spatial(frame[row], int(window))[0]  for row in range(rows)]) for window in windows]

diff_box_values_y = [np.array([diff_box_spatial(frame[:, col], int(window))[0] for col in range(cols)]) for window in windows]