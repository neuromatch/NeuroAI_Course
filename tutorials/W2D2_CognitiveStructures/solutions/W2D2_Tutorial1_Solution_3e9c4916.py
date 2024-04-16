
"""
Discussion: How would you provide intuitive reasoning or rigorous mathematical proof behind the fact that random high-dimensional vectors (note that each of the components is drawn from uniform distribution with zero mean) approximately orthogonal?

Observe that as each of the components are independent and they are sampled from distribution with zero mean, it means that expected value of dot product E(x*y) = E(\sum_i x_i * y_i) = (linearity of expectation) \sum_i E(x_i * y_i) = (independence) \sum_i (E(x_i) * E(y_i)) = 0.
""";