
"""
Discussion: What is the qualitative difference between trajectories propagation through these networks? Does it fit what we have seen earlier with wide student approximation?

Indeed, a standard network (with sigma = 2) is much more expressive, it folds the space here and there, creating vivid anf tangled representation with each additional layer, whereas the quasi-linear network preserves the original structure.
It is in line with the experiments on wide student approximation as shallow and wide networks cannot express the tangled representation which a standard net creates.
"""