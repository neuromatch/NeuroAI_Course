
"""
Discussion: Why do we need to normalize the vector obtained in the result of bundling operation? What length do you expect to receive without normalization?

We would like to preserve unitary length of the vector so it fits the rules of the vector space we've defined. If we simply add three vectors together we can calculate the resulted length by taking dot product with itself - it will be the sum of pairwise dot products of all vectors in the sum (with repetition of vectors with themselves), thus the sum is going to be around three (remember that <x, y> = 0 while <x, x> = 1), meaning that length of obtained vector is sqrt(3).
""";