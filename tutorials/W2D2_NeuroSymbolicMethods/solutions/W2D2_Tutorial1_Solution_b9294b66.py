
"""
Discussion: Why do we need to normalize the vector obtained as a result of the bundling operation? What length do you expect to receive without normalization?

We would like to preserve the unitary length of the vector so it fits the rules of the vector space we've defined. If we simply add three vectors together, we can calculate the resulted length by taking the dot product with itself - it will be the sum of pairwise dot products of all vectors in the sum (with repetition of vectors with themselves), thus the sum is going to be around three (remember that <x, y> = 0 while <x, x> = 1), meaning that length of the obtained vector is sqrt(3).
""";