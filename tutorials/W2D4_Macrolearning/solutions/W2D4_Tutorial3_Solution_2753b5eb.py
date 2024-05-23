
def finetune(model, x_finetune, y_finetune, finetune_gradient_steps):
    """
    Take a fixed number of gradient steps for the given x_finetune and y_finetune.

    Inputs:
    - model (MetaLearningModel): trained meta learning model.
    - x_finetune (torch.tensor): features (days) of the specific task.
    - y_finetune (torch.tensor): outcomes (prices) of the specific task.
    - finetune_gradient_steps (int): number of gradient steps to perform for this task.
    """
    #apply normalization on days
    x_finetune = (x_finetune - model.mean) / model.std

    #need to create clone, so that we preserve meta-learnt parameters
    clone = model.deep_clone_model(model.model)
    optimizer = optim.SGD(clone.parameters(), lr = model.inner_learning_rate)

    for _ in range(finetune_gradient_steps):
        optimizer.zero_grad()
        loss = model.loss_fn(clone(x_finetune), y_finetune)
        loss.backward()
        optimizer.step()

    return clone