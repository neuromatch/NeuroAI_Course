
"""
Question: For the standard clean images, how do the RDMs change across the model layers, how do they compare to the category structure, and why?

For clean images representing the same digit, their representations in the deeper layers of the network are remarkably similar and align well with the inherent category structure. This manifests as a block diagonal structure. In contrast, this block effect is less pronounced in the earlier layers of the network. The initial layers focus more on capturing general and granular visual features. This progression from generic to more refined feature extraction across layers underscores the hierarchical nature of learning in deep neural networks, where complex representations are built upon the simpler ones extracted at earlier stages.
""";