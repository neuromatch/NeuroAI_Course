"""
When one of the arms has a high reward, it is easier to identify it. The uncertainty of
the agent thus reduces quickly, and they spend less time in exploration. When the
probabilities of rewards are more equal (close to 0.5), many trials are needed to
reduce the uncertainty, and the exploration phase lasts a long time.
"""