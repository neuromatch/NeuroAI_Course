
set_seed(42)

def train_model(net, X, y, n_epochs, lr, progressbar=True):
    """
    Perform training of the network.

    Inputs:
    - net (nn.Sequential): network.
    - X (torch.tensor): input data.
    - y (torch.tensor): output data.
    - n_epochs (int): number of epochs to train the model for.
    - lr (float): learning rate for optimizer (we will use `Adam` by default).
    - progressbar (bool, default = True): whether to use additional bar for displaying training progress.

    Outputs:
    - Es (np.ndarray): array which contains loss for each epoch.
    """

    # Set up optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)

    # Run training loop
    Es = np.zeros(n_epochs)
    for n in (tqdm(range(n_epochs)) if progressbar else range(n_epochs)):
        y_pred = net(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Es[n] = float(loss.detach())

    return Es

Es = train_model(net, X, y, 10, 1e-3)
Es[0]