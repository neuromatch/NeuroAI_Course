"""
Since our network is untrained, there isn't much difference in the cosine similarities
within and across image classes. This lack of clear structure in the similarity matrix
is expected at this stage because the network has not yet learned to distinguish between
different classes.

Ideally, we should observe a very high cosine similarity for images within the same
class (along the diagonal) and very low cosine similarity for images from different
classes (off-diagonal).
"""