def generate_trajectory(model, inputs, device):
    inputs = inputs.to(device)
    batch_size = inputs.size(0)
    h = model.init_hidden(batch_size).to(device)

    loss = 0
    outputs = []
    hidden_states = []
    with torch.no_grad():
        for t in range(inputs.shape[1]):
            # Forward the model's input and hidden state to obtain the model
            # output and hidden state *h*.
            # Note that you should index the input tensor by the time dimension
            # Capture any additional outputs in 'rest'
            output, h, *rest = model(inputs[:, t], h)
            outputs.append(output)
            hidden_states.append(h.detach().clone())

    return torch.stack(outputs, axis=1).to(device), torch.stack(hidden_states, axis=1).to(device)