"""
- Inpainting, similar to autoencoding, necessitates a substantial amount of data for effective training.
- Even after 10,000 examples and 10 epochs, the network is still learning. The multiple masked images
  generated from the same image means the network keeps learning from the same images.
"""