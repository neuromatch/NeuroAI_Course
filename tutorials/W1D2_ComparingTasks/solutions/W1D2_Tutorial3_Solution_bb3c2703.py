"""
The adaptation to fast-changing latent variables, as described in the changes in activation patterns over trials,
shows the algorithm's capability to adjust its internal representations based on feedback from the environment.

- The first and last plots show very fast convergence towards the optimal arm when the reward probabilities are
very different. The trajectory moves rapidly towards one or the other side of the PC space.

- In the middle two plots, correspond to more difficult settings, the model takes longer to converge,
The model starts by exploiting the left arm, but after sampling a right action, ends up converging on the right.

- The first PC appears to correspond to the certainty of the algorithm about the optimal arm
(low PC value = left arm, high PC value = right arm).
"""