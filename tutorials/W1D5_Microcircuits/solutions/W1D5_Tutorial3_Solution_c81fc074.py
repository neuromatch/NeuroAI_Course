class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, T: int, dm: int, dk: int):
        """
        Scaled Dot Product Attention
        Args:
            T (int): context length
            dm (int): model dimension
            dk (int): key dimension
        Note:
            we assume dm == dv
        """
        super().__init__()
        self.T = T  # context length
        self.dm = dm  # model dimension
        self.dk = dk  # key dimension
        self.scale = 1.0 / math.sqrt(dk)

        # positional Encoding
        self.position = PositionalEncoding(T, dm)

        # self-attention layers
        self.Wq = torch.nn.Linear(dm, dk, bias=False)  # query layer
        self.Wk = torch.nn.Linear(dm, dk, bias=False)  # key layer
        self.Wv = torch.nn.Linear(dm, dm, bias=False)  # value layer

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): input tensor of shape (T, d)
        """
        # Positional Encoding
        x = x + self.position()

        # (Scaled Dot-Product Attention)
        Q = self.Wq(x)  # Query
        K = self.Wk(x)  # Key
        V = self.Wv(x)  # Value
        QK = Q @ K.T  # Query Key product
        S = QK * self.scale  # Scores (scaled against saturation)
        S_softmax = torch.softmax(S, dim=-1)  # softmax attention scores (row dimensions)
        A = S_softmax @ V  # scaled dot-product attention
        return A