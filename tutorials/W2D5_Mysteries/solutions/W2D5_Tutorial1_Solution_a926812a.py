class SecondOrderNetwork(nn.Module):
    def __init__(self, use_gelu):
        super(SecondOrderNetwork, self).__init__()
        # Define a linear layer for comparing the difference between input and output of the first-order network
        self.comparison_layer = nn.Linear(100, 100)

        # Linear layer for determining wagers, mapping from 100 features to a single output
        self.wager = nn.Linear(100, 1)

        # Dropout layer to prevent overfitting by randomly setting input units to 0 with a probability of 0.5 during training
        self.dropout = nn.Dropout(0.5)

        # Select activation function based on the `use_gelu` flag
        self.activation = torch.relu

        # Additional activation functions for potential use in network operations
        self.sigmoid = torch.sigmoid

        self.softmax = nn.Softmax()

        # Initialize the weights of the network
        self._init_weights()

    def _init_weights(self):
        # Uniformly initialize weights for the comparison and wager layers
        init.uniform_(self.comparison_layer.weight, -1.0, 1.0)
        init.uniform_(self.wager.weight, 0.0, 0.1)

    def forward(self, first_order_input, first_order_output):
        # Calculate the difference between the first-order input and output
        comparison_matrix = first_order_input - first_order_output

        #Another option is to directly calculate the per unit MSE to use as input for the comparator matrix
        #comparison_matrix = nn.MSELoss(reduction='none')(first_order_output, first_order_input)

        # Pass the difference through the comparison layer and apply the chosen activation function
        comparison_out=self.dropout(self.activation(self.comparison_layer(comparison_matrix)))

        # Calculate the wager value, applying dropout and sigmoid activation to the output of the wager layer
        wager = self.sigmoid(self.wager(comparison_out))

        return wager