
"""
Discussion: Can you provide an intuitive reason, or rigorous mathematical proof, behind the fact that random high-dimensional vectors are approximately orthogonal? Here each of the components of the vectors is drawn from a independent and identical distribution, e.g. a normal distribution.

Observe that as each of the components is independent and they are sampled from a distribution with zero mean, it means that the expected value of dot product E(x*y) = E(\sum_i x_i * y_i) = (linearity of expectation) \sum_i E(x_i * y_i) = (independence) \sum_i (E(x_i) * E(y_i)) = 0.
""";