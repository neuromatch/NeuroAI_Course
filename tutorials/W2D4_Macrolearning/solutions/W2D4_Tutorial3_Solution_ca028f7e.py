
class MetaLearningModel(UtilModel):
    def __init__(self, *args, **kwargs):
        """ Implementation of MAML (Finn et al. 2017).
        """
        super().__init__(*args, **kwargs)

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.outer_learning_rate)

    def inner_loop(self, x_inner, y_inner):
        """ Compute parameters of the model in the inner loop but not update model parameters.
        """
        #calculate loss for inner data
        loss = self.loss_fn(self.model(x_inner), y_inner)

        #inner data gradients w.r.t calculated loss
        grads = torch.autograd.grad(loss, self.model.parameters())

        #update weights based on inner data gradients; observe that we don't update `self.model.parameters()` here
        parameters = [parameter - self.inner_learning_rate * grad for parameter, grad in zip(self.model.parameters(), grads)]

        return parameters

    def outer_loop(self, x_outer, y_outer, weights):
        """ Compute loss for outer dats with already calculated parameters on inner data.
        """
        #observe that parameters come from inner data, while loss is calculated on outer one; it is the heart of meta-learning approach
        return self.loss_fn(self.manual_output(weights, x_outer), y_outer)

    def train_tasks(self, tasks):
        """ Utility method to train an entire epoch in one go. Implements the meta-update step for a batch of tasks.
        """
        #prepare for loss accumulation
        metaloss = 0
        self.optimizer.zero_grad()

        #take inner and outer data from dataset
        x_inner, y_inner, x_outer, y_outer = tasks

        #apply normalization on days
        x_inner = (x_inner - self.mean) / self.std
        x_outer = (x_outer - self.mean) / self.std

        #for each task there is going to be inner and outer loop
        for task_idx in range(x_inner.shape[0]):

            #find weights (line [6])
            parameters = self.inner_loop(x_inner[task_idx].type(torch.float32).to(device), y_inner[task_idx].type(torch.float32).to(device))

            #find meta loss w.r.t to found weights in inner loop (line [9])
            task_metaloss = self.outer_loop(x_outer[task_idx].type(torch.float32).to(device), y_outer[task_idx].type(torch.float32).to(device), parameters)

            #contribute to metaloss
            metaloss += task_metaloss

        #update model parameters
        metaloss.backward()
        self.optimizer.step()

        return metaloss.item() / x_inner.shape[0]