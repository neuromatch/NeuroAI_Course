"""
- For this task, having fewer than 1000 training points is insufficient for the model to learn effectively.
The limited number of examples does not provide enough variability and information for the model to generalize
well to new, unseen data.

- When the dataset size is increased to 10,000 training points, the network reaches a steady-state test
performance after just one iteration through the dataset. This indicates that a larger dataset allows the
model to learn more efficiently and achieve stable performance with minimal passes through the data.

- A dataset of 1000 training points can also lead to good performance, but it requires approximately 10
iterations through the dataset to reach that point. In total, the classifier sees about 10,000 images,
but many of these images are repeated. This implies that while the diversity of 1000 unique images is
sufficient, the model needs multiple exposures to the same images to achieve the same performance level
as with a larger, more varied dataset.
"""