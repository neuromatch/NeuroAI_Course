"""
RIMs are built as a collection of modules, each operating independently. They don't
continuously interact with each other; instead, they primarily operate separately and
only occasionally connect through attention. This attention mechanism allows a module to
focus on specific, relevant parts of the input data when necessary.

Because each module in a RIM can focus on learning different features or aspects of the
data, it becomes very adaptable. For instance, one module might become specialized in
recognizing edges, another in textures, and so on. When the model encounters a new
environment or different image sizes, like 19x19 or 24x24, each module uses its
specialized knowledge to handle the changes in its specific area. This modular and
focused approach allows RIMs to maintain performance across varied conditions,
demonstrating out-of-distribution generalization.
"""