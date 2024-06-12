
"""
Discussion: Does the amount of covariate shift impact the model's performance? What happens at the borders of the training period—does the model still capture the dynamics right before and after it?

Indeed, the bigger the covariate shift (the more distinct the days are), the worse the performance we observe. In both border cases, the model performs poorly; what is more - even on the fraction of training data near these regions, we can observe that the model is going to lose the desired dynamics.
""";