# below, define diff_x and diff_y
diff_x = np.diff(frame, axis = 1)
diff_y = np.diff(frame, axis = 0)

plot_spatial_diff(frame, diff_x,  diff_y)