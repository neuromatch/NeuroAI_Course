"""
At the level of perceptual states W, there is a substantial asymmetry in the KL-divergence expected when the
model says ‘seen’ vs. ‘unseen’ (lefthand panel). This is due to the large belief updates invoked in the
perceptual layer W by samples that deviate from the lower lefthand corner - from absence. In contrast, when
we compute KL-divergence for the A-level (righthand panel), the level of prediction error is symmetric across
seen and unseen decisions, leading to "hot" zones both at the upper righthand (present) and lower lefthand
(absent) corners of the 2D space.

Intuitively, this means that at the W-level, there's a noticeable difference in the KL-divergence values
between "seen" and "unseen" predictions. This large difference is mainly due to significant updates in the
model's beliefs at this level when the detected samples are far from what is expected under the condition of
"absence." However, when we analyze the K-L divergence at the A-level, the discrepancies in prediction errors
between "seen" and "unseen" are balanced. This creates equally strong responses in the model, whether something
is detected or not detected.

We can also sort the KL-divergences as a function of whether the model "reported" presence or absence. As
can be seen in the bar plots below, there is more asymmetry in the prediction error at the W compared to the
A levels.

"""