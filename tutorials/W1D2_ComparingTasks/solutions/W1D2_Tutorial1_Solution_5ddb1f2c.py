
def cost_regression(output, target):
    criterion = nn.MSELoss()
    cost = criterion(output, target)
    return cost