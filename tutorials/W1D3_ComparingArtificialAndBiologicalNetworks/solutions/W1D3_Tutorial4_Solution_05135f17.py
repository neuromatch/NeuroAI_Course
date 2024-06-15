
"""
Discussion: 1. Does the amount of distortion after projection depend on the dimension $d$ of the original space? Observe the dimension $k$ that preserves Euclidean distance up to a small distortion for both the 2-neuron and 100-neuron datasets.

2. What is the distance between two identical stimuli after random projection?

1. No. Empirically, the dimension that preserves Euclidean distance up to a small distortion for the 100-neuron dataset is similar to the 2-neuron dataset. Theoretically, the distortion bound is independent of the original dimension (https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).

2. The distance is always 0.
""";