
def cost_classification(output, target):
    criterion = nn.BCELoss()
    target = target.to(torch.int64)
    batch_size = target.shape[0]
    # Create a tensor of zeros with shape (batch_size, num_classes)
    target_one_hot = torch.zeros((batch_size, 10), device=target.device)
    # Use scatter_ to place '1's in the correct indices
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    cost = criterion(output, target_one_hot)
    return cost