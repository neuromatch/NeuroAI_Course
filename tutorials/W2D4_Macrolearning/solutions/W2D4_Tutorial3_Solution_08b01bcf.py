
class FruitSupplyDataset(Dataset):
    def __init__(self, num_epochs = 1, num_tasks = 1, num_samples = 1, days = days, also_sample_outer = True):
        """Initialize particular instance of `FruitSupplyDataset` dataset.

        Inputs:
        - num_epochs (int): Number of epochs the model is going to be trained on.
        - num_tasks (int): Number of tasks to sample for each epoch (the loss and improvement is going to be represented as sum over considered tasks).
        - num_samples (int): Number of days to sample for each task.
        - days (np.ndarray): Summer and autumn days to sample from.
        - also_sample_outer (bool): `True` if we want to sample inner and outer data (necessary for training).

        Raises:
        - ValueError: If the number of sampled days `num_samples` exceeds number of days to sample from.
        """

        if also_sample_outer:
            if num_samples > days.shape[0] // 2:
                raise ValueError("Number of sampled days for one task should be less or equal to the total amount of days divided by two as we sample inner and outer data.")
        else:
            if num_samples > days.shape[0]:
                raise ValueError("Number of sampled days for one task should be less or equal to the total amount of days.")

        #total amount of data is (2/4 x num_epochs x num_tasks x num_samples) (2/4 because -> x_inner, x_outer, y_inner, y_outer; outer is optional)
        self.num_epochs = num_epochs
        self.num_tasks = num_tasks
        self.num_samples = num_samples
        self.also_sample_outer = also_sample_outer
        self.days = days

    def __len__(self):
        """Calculate the length of the dataset. It is obligatory for PyTorch to know in advance how many samples to expect (before training),
        thus we enforced to icnlude number of epochs and tasks per epoch in `FruitSupplyDataset` parameters."""

        return self.num_epochs * self.num_tasks

    def __getitem__(self, idx):
        """Generate particular instance of task with prefined number of samples `num_samples`."""

        A = np.random.uniform(min_A, max_A, size = 1)
        B = np.random.uniform(min_B, max_B, size = 1)
        phi = np.random.uniform(min_phi, max_phi, size = 1)
        C = np.random.uniform(min_C, max_C, size = 1)

        #`replace = False` is important flag here as we don't want repeated data
        inner_sampled_days = np.expand_dims(np.random.choice(self.days, size = self.num_samples, replace = False), 1)

        if self.also_sample_outer:

            #we don't want inner and outer data to overlap
            outer_sampled_days = np.expand_dims(np.random.choice(np.setdiff1d(self.days, inner_sampled_days), size = self.num_samples, replace = False), 1)

            return inner_sampled_days, A * inner_sampled_days ** 2 + B * np.sin(np.pi * inner_sampled_days + phi) + C, outer_sampled_days, A * outer_sampled_days ** 2 + B * np.sin(np.pi * outer_sampled_days + phi) + C

        return inner_sampled_days, A * inner_sampled_days ** 2 + B * np.sin(np.pi * inner_sampled_days + phi) + C

    def sample_particular_task(self, A, B, phi, C, num_samples):
        """Samples for the particular instance of the task defined by the tuple of parameters (A, B, phi, C) and `num_samples`."""

        sampled_days = np.expand_dims(np.random.choice(self.days, size = num_samples, replace = False), 1)
        return sampled_days, A * sampled_days ** 2 + B * np.sin(np.pi * sampled_days + phi) + C