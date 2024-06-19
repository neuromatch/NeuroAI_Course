
"""
Discussion: 1. Why do you think the filter is asymmetric?
2. How might a filter influence the sparsity patterns observed in data?

1. As the filter reflects how we process time series data, it accounts for the past & for the future with different powers. Particularly, for this filter, past information is not included in the result at all, while the future one is evenly distributed for the defined window size.
2. Note that the filter takes the average of the future (not only one point of time). Thus, it would be much smoother than the regular one.
""";