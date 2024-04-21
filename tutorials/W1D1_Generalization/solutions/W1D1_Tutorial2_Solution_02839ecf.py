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

trajectory, hidden_states = generate_trajectory(model,
                                             inputs[0].unsqueeze(0),
                                             device)

with plt.xkcd():
  plot_hidden_unit_activations(hidden_states=hidden_states.squeeze().detach().cpu().numpy(),
                               times=times,
                               neurons_to_plot=7,
                               title='Hidden units')
  plot_muscles_over_time(times, trajectory.squeeze().detach().cpu().numpy(), 'Generated muscle activity')