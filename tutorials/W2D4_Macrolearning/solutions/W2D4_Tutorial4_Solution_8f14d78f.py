
"""
Discussion: What should be changed in the implementation approach (code base) to reflect Darwinian and Lamarckian evolutions?

Observe that for Baldwin effect, while applying tournament paradigm, we select for individuals which learn faster by making them parents for the offsprings.
For Darwinian evolution, we might want to not select at all (remove `top` search of the tournament group).
For Lamarckian evolution, we might want to change parameters of the agent each time it is exposed to the new task (meaning we don't need to create copy of the agent at all).""";