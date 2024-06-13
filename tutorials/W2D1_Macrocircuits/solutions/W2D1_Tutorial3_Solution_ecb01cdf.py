class ModularActor(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM):
        """
        Initializes the modular actor model with given dimensions.

        Inputs:
        - OBS_DIM (int): The dimension of the observation input.
        - ACTION_DIM (int): The dimension of the action output.
        - TARGET_DIM (int): The dimension of the target input.

        Outputs:
        - None
        """
        super().__init__()
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM
        self.RNN_SIZE = 128  # RNN hidden size
        MLP_SIZE = 300  # number of neurons in one MLP layer

        self.rnn = nn.LSTM(input_size=OBS_DIM + ACTION_DIM + TARGET_DIM, hidden_size=self.RNN_SIZE)
        self.l1 = nn.Linear(self.RNN_SIZE, MLP_SIZE)
        self.l2 = nn.Linear(MLP_SIZE, MLP_SIZE)
        self.l3 = nn.Linear(MLP_SIZE, ACTION_DIM)

    def forward(self, x, hidden_in):
        """
        Computes the action based on the current input and hidden state.

        Inputs:
        - x (tensor): The current input to the model, which includes observation, action, and target information.
        - hidden_in (tuple): The initial hidden state for the LSTM.

        Outputs:
        - a (tensor): The action output from the model.
        - hidden_out (tuple): The updated hidden state from the LSTM.
        """
        #######################################################
        # TODO: Pass 'x' to the MLP module, which consists of two linear layers with ReLU nonlinearity.
        # First, pass 'x' to the first linear layer, 'self.l1', followed by 'F.relu'.
        # Second, pass 'x' again to the second linear layer, 'self.l2', followed by 'F.relu'.
        #######################################################
        x, hidden_out = self.rnn(x, hidden_in)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        a = torch.tanh(self.l3(x))

        return a, hidden_out