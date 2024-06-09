"""
The agent models the reward probabilities of the two arms using fixed distributions.
There is no mechanism by which the agent can detect that there's a change in the
environment and rapidly update its beliefs. Its underlying model is not flexible,
and it cannot adapt to changes in the environment, leading to suboptimal performance.
"""