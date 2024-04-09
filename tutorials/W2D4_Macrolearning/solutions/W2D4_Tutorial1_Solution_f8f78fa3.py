
"""
Discussion: Does the power of covariate shift impact the model performance? What happens on the borders of the training period, does model still capture the fraction of dynamics right before/after it?

Indeed, the bigger the power of covariate shift (the more distinct the days are), the worse performance we observe. In both border-cases model performs poorly, what is more - even on the fraction of training data near these regions we can observe that model is going to lose the desired dynamics.
""";