
"""
Discussion: 1. Why do you think the filter is assymetric?
2. How might a filter influence the sparsity patterns observed in data?

1. As filter reflects how do we process time series data, it accounts for the past & for the future with different power. Particularly, for this filter past information is not included in the result at all while the future one is evenly distributed for the defined window size.
2. Note that it the filter takes average of the future (not only one time point), thus it would be much smoother than the regular one.
""";