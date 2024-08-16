
def compute_loss(net, X, y):
    """
    Calculate loss on given network and data.

    Inputs:
    - net (nn.Sequential): network.
    - X (torch.tensor): input data.
    - y (torch.tensor): output data.

    Outputs:
    - loss (float): computed loss.
    """
    loss_fn = nn.MSELoss()

    y_pred = net(X)
    loss = loss_fn(y_pred, y)
    loss = float(loss.detach())
    return loss

loss = compute_loss(net, X, y)
np.testing.assert_allclose(loss, 0.0, err_msg = "Expected value of loss is different!", atol = 1e-3)