
def dcl_loss(pos_pairs, neg_pairs, indices_tuple, temperature=0.07):
    """
    Computes the Decoupled Contrastive Learning loss.

    Returns:
        torch.Tensor: The computed loss value.
    """
    a1, _, a2, _ = indices_tuple  # Unpack indices

    if len(a1) == 0 or len(a2) == 0:
        return 0

    dtype = neg_pairs.dtype
    pos_pairs = pos_pairs.unsqueeze(1) / temperature  # Scale positive pairs by temperature
    neg_pairs = neg_pairs / temperature  # Scale negative pairs by temperature
    n_per_p = to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)  # Indicator matrix for matching pairs
    neg_pairs = neg_pairs * n_per_p  # Zero out non-matching pairs
    neg_pairs[n_per_p == 0] = neg_inf(dtype)  # Replace non-matching pairs with negative infinity

    # Compute the maximum value for numerical stability
    max_val = torch.max(
        pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
    ).detach()
    # Compute numerator and denominator for the loss
    numerator = torch.exp(pos_pairs - max_val).squeeze(1)
    denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1)
    log_exp = torch.log((numerator / denominator) + small_val(dtype))
    return -log_exp  # Return the negative log of the exponential

def pair_based_loss(similarities, indices_tuple, lossfunc):
    """
    Computes pair-based loss using the provided loss function.

    Args:
        similarities : torch.Tensor
            A tensor of pairwise similarities. For n_examples, the shape should be
            (n_examples, n_examples).
        indices_tuple : tuple
            A tuple of indices for positive and negative pairs. The tuple should
            contain 4 tensors: a1, p, a2, n. The tensors a1 and p contain indices
            for positive pairs, while a2 and n contain indices for negative pairs.
            a1 and p should have the same length, and a2 and n should have the same
            length. a1[i] and p[i] should form a positive pair, such that they have the
            same label. Similarly, a2[i] and n[i] should form a negative pair, such that
            they have different labels.
        lossfunc : function
            The loss function to be applied for computing the loss.
    """
    # Computes pair-based loss using the provided loss function
    a1, p, a2, n = indices_tuple  # Unpack indices
    pos_pair, neg_pair = [], []
    if len(a1) > 0:
        pos_pair = similarities[a1, p]  # Extract positive pairs
    if len(a2) > 0:
        neg_pair = similarities[a2, n]  # Extract negative pairs
    return lossfunc(pos_pair, neg_pair, indices_tuple)  # Apply loss function

# Number of epochs for training
epochs = 10

# Automatically select the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output the device that will be used
print(f"Using device: {device}")

# Load the MNIST training dataset with the defined transformations
train_dset = torchvision.datasets.MNIST("./", train=True, transform=mnist_transforms)
train_loader = DataLoader(train_dset, batch_size=50, shuffle=True)  # Enable persistent_workers=True if more than 1 worker to save CPU

# Cleanup: delete the network and free up memory if this block is re-run
try:
    del mynet
    gc.collect()
    torch.cuda.empty_cache()
except:
    pass

# Initialize the model with specified input, output, and hidden dimensions
mynet = Model(in_dim=784, out_dim=128, hidden_dim=256)
_ = mynet.to(device)  # Move the model to the selected device

# Enable training mode, which may affect dropout and other layers
mynet.train(mode=True)
print("Is the network in training mode?", mynet.training)

# Initial learning rate and decay factor for the optimizer
init_lr = 3e-4
lr_decay_factor = 0.5

# Initialize the optimizer with model parameters and learning rate
optimizer = torch.optim.AdamW(mynet.parameters(), lr=init_lr, weight_decay=1e-2)

# Tracker to keep track of loss values during training
loss_tracker = []

# Training loop over the specified number of epochs
for epoch_id in range(1, epochs+1):
    loss_epoch_tracker = 0
    batch_counter = 0

    # Adjust learning rate for the current epoch
    new_lrate = init_lr * (lr_decay_factor ** (epoch_id / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate

    batches_in_epoch = len(train_loader)
    for data_batch in train_loader:
        optimizer.zero_grad()  # Zero out gradients

        # Get images and labels from the batch
        train_img, train_label = data_batch
        batch_size = train_img.shape[0]

        # Flatten images and move data to the selected device
        flat = train_img.reshape(batch_size, -1).to(device, non_blocking=True)
        train_label = train_label.to(device, non_blocking=True)

        # Forward pass through the network
        predicted_results = mynet(flat)

        # Compute cosine similarity matrix for the batch
        similarities = cos_sim(predicted_results)

        # Get pairs of indices for positive and negative pairs
        label_pos_neg = get_all_pairs_indices(train_label)

        # Compute the loss using the decoupled contrastive learning loss function
        final_loss = torch.mean(pair_based_loss(similarities, label_pos_neg, dcl_loss))

        # Compute gradients from the loss
        final_loss.backward()

        # Update the model parameters using the optimizer
        optimizer.step()

        # Convert the loss to a single CPU scalar
        loss_cpu_number = final_loss.item()

        # Keep track of the losses for visualization
        loss_epoch_tracker += loss_cpu_number
        batch_counter += 1

        # Print the current epoch, batch number, and loss every 500 batches
        if batch_counter % 500 == 0:
            print("Epoch {}, Batch {}/{}, loss: {}".format(epoch_id, batch_counter, batches_in_epoch, loss_cpu_number))

    # Print the average loss for the epoch
    print("Epoch average loss {}".format(loss_epoch_tracker / batch_counter))