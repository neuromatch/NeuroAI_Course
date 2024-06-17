
"""
Discussion: Why do you think the bundled representation is superior for the Rastrigin function?

The Rastrigin function is a superposition of independent functions of the input variable dimensions. The bundled representation is a superposition of a high-dimensional representation of the input dimensions, making it easier to learn this function, which is additive. For the bound representation, we have to learn a mapping from each tuple of input values to the appropriate output value, meaning more samples are required to approximate the function.
""";