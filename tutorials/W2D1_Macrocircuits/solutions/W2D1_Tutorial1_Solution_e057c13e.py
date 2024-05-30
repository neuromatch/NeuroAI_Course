
"""
Discussion: What is the qualitative difference between trajectories propagation through these networks? Does it fit what we have seen earlier with wide student approximation?

Indeed, standard net (with sigma = 2) is much more expressive, it folds the space here and there, creating vivid anf tangled representation with each additional layer, whereas quasi-linear network preserves the original structure.
It is inline with the experiments on wide student approximation as wide nets (but not deep) cannot express tangled representation which standard net creates.
""";