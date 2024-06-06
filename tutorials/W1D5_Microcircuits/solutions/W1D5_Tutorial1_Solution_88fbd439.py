def hard_thres(frame, theta):
    """
    Return hard thresholded array of values based on the parameter value theta.

    Inputs:
    - frame (np.array): 2D signal.
    - theta (float, default = 0): threshold parameter.
    """
    frame_HT = frame.copy()
    frame_HT[frame_HT < theta] = 0
    return frame_HT