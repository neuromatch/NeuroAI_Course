
"""
Discussion: Which of the training schedulers worked better for this parituclar example holding all hyperparameters values fixed (observe that number of epochs for sequential training is doubled in sequential training mode so the model observes the same amount of the data as in the intersepersed one)? Why do you think it happened that way?

In the sequential training, model is constantly shifting from learning one type of relation to another which is basically what we have tried during the first section of the tutorial; still, here it makes sense because we change data source for each epoch which makes the model to replay previous data. We will expand on this notion in Tutorial 4.
""";