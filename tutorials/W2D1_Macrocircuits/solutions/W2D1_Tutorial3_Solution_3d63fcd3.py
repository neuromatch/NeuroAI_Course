class HolisticActor(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM):
        """
        Initializes the holistic actor model with given dimensions.

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
        self.RNN_SIZE = 220  # RNN hidden size

        self.rnn = nn.LSTM(input_size=OBS_DIM + ACTION_DIM + TARGET_DIM, hidden_size=self.RNN_SIZE)
        self.l1 = nn.Linear(self.RNN_SIZE, ACTION_DIM)

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
        # TODO: Pass the input 'x' and the previous hidden state 'hidden_in' to the RNN module 'self.rnn'.
        # Get the output 'x' and the hidden state 'hidden_out' from the RNN module.
        # Refer to https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html.
        # Hint: 'self.rnn' takes two arguments as inputs and outputs two things.
        # The first position corresponds to 'x', and the second position corresponds to the hidden state.
        #######################################################
        x, hidden_out = self.rnn(x, hidden_in)

        a = torch.tanh(self.l1(x))
        return a, hidden_out