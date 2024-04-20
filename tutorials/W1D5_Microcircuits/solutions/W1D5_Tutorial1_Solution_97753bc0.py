low_perc = np.percentile(frame, 80)
frame_HT = hard_thres(frame, low_perc)
plot_images([frame, frame_HT])