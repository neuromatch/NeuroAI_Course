"""
Loose definition:

The Omniglot task you just tried has one labelled example per class: the reference
image defines the reference class. People tested on the Omniglot task generally
display far higher performance than chance (here, 11%). Based on this, we often say
(loosely) that people display a sample complexity N=1 on this task.

Strict definition:

When I tried this task 20 times, I got 18 correct answers. Based on that, I can
estimate a 90% confidence interval (delta = 0.05) for my error rate using binomial
statistics as [.02, .23]. Thus, I can state that my sample complexity for this task
is N(epsilon=.1, delta=.23) = 1.

My numbers are on the low end of what has been demonstrated in the literature.
Lake et al. (2015) estimated an error rate in humans of 4.5% with 20 distractors.
"""