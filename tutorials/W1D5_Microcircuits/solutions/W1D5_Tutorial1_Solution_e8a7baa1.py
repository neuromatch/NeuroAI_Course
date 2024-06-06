def ReLU(x, theta = 0):
    """
    Calculates ReLU function for the given level of theta.

    Inputs:
    - x (np.ndarray): input data.
    - theta (float, default = 0): threshold parameter.

    Outputs:
    - thres_x (np.ndarray): filtered values.
    """

    thres_x = np.maximum(x - theta, 0)

    return thres_x