
def inspect_decoder(model):
    """
    Inspect decoder to verify that it processes inputs in the expected way.

    Args:
        model: the TrOCR model
    """
    # Create an empty tensor (batch size of 1) to feed it to the encoder.
    # Remember that images should have 3 channels and have size 384x384
    # Recall that images are fed in pytorch with tensors of shape
    # batch x channels x height x width
    single_input = torch.zeros(1, 3, 384, 384).to(device)

    # Run the input through the encoder.
    output = model.encoder(single_input)

    # Measure the number of hidden tokens which are the output of the encoder
    hidden_shape = output['last_hidden_state'].shape

    assert hidden_shape[0] == 1
    assert hidden_shape[1] == 577
    assert hidden_shape[2] == 768