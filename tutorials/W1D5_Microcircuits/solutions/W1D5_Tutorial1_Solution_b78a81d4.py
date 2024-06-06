def diff_box_spatial(data, window, pad_size = 4):
    """
    Implement & apply spatial filter on the signal.

    Inputs:
    - data (np.ndarray): input signal.
    - window (int): size of the window.
    - pad_size (int, default = 4): size of pad around data.
    """
    filter = np.concatenate([np.repeat(0, pad_size), np.repeat(-1, window), np.array([1]), np.repeat(-1, window), np.repeat(0, pad_size)]).astype(float)

    #normalize
    filter /= np.sum(filter**2)**0.5

    #make sure the filter sums to 0
    filter_plus_sum =  filter[filter > 0].sum()
    filter_min_sum = np.abs(filter[filter < 0]).sum()
    filter[filter > 0] *= filter_min_sum/filter_plus_sum

    #convolution of the signal with the filter
    diff_box = np.convolve(data, filter, mode='full')[:len(data)]
    diff_box[:window] = diff_box[window]
    return diff_box, filter